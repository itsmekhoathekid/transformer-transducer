import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import torchaudio
import torchaudio.transforms as T
from speechbrain.lobes.features import Fbank
import speechbrain as sb

# [{idx : {encoded_text : Tensor, wav_path : text} }]


def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    import json

    with open(path, "r", encoding= 'utf-8') as f:
        data = json.load(f)
    return data

class Vocab:
    def __init__(self, vocab_path):
        self.vocab = load_json(vocab_path)
        self.itos = {v: k for k, v in self.vocab.items()}
        self.stoi = self.vocab

    def get_sos_token(self):
        return self.stoi["<s>"]
    def get_eos_token(self):
        return self.stoi["</s>"]
    def get_pad_token(self):
        return self.stoi["<pad>"]
    def get_unk_token(self):
        return self.stoi["<unk>"]
    def __len__(self):
        return len(self.vocab)





class Speech2Text(Dataset):
    def __init__(self, json_path, vocab_path, config, apply_spec_augment=False):
        super().__init__()
        self.data = load_json(json_path)
        self.vocab = Vocab(vocab_path)
        self.sos_token = self.vocab.get_sos_token()
        self.eos_token = self.vocab.get_eos_token()
        self.pad_token = self.vocab.get_pad_token()
        self.unk_token = self.vocab.get_unk_token()
        self.apply_spec_augment = apply_spec_augment
        
        freq_width = int(0.15 * config['fbank']['n_mels'])  # Tính toán độ rộng tần số

        self.augment = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_width),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_width),  # 2 mask
            torchaudio.transforms.TimeMasking(time_mask_param=config['fbank']['n_mels'], p=1.0),
            torchaudio.transforms.TimeMasking(time_mask_param=config['fbank']['n_mels'], p=1.0)         # 2 mask
        )

        self.fbank = Fbank(
            sample_rate=config['fbank']['sample_rate'],
            n_mels=config['fbank']['n_mels'],
            n_fft=config['fbank']['n_fft'],
            win_length=config['fbank']['win_length'],
            hop_length=config['fbank']['hop_length'],
        )

    def __len__(self):
        return len(self.data)

    def get_fbank(self, waveform):
        fbank = self.fbank(waveform)
        if self.apply_spec_augment:
            fbank = self.augment(fbank)
        return fbank.squeeze(0)  # [T, 80]


    def extract_from_path(self, wave_path):
        sig  = sb.dataio.dataio.read_audio(wave_path)
        return self.get_fbank(sig.unsqueeze(0))

    def __getitem__(self, idx):
        current_item = self.data[idx]
        wav_path = current_item["wav_path"]
        encoded_text = torch.tensor(current_item["encoded_text"] + [self.eos_token], dtype=torch.long)
        decoder_input = torch.tensor([self.sos_token] + current_item["encoded_text"] + [self.pad_token], dtype=torch.long)
        fbank = self.extract_from_path(wav_path).float()  # [T, 512]

        return {
            "text": encoded_text,
            "fbank": fbank,
            "text_len": len(encoded_text),
            "fbank_len": fbank.shape[0],
            "decoder_input": decoder_input,
        }
    
from torch.nn.utils.rnn import pad_sequence

def calculate_mask(lengths, max_len):
    """Tạo mask cho các tensor có chiều dài khác nhau"""
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask

def causal_mask(batch_size, size):
    """Tạo mask cho decoder để tránh nhìn thấy tương lai"""
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, T]

def speech_collate_fn(batch):
    decoder_outputs = [torch.tensor(item["decoder_input"]) for item in batch]
    texts = [item["text"] for item in batch]
    fbanks = [item["fbank"] for item in batch]
    text_lens = torch.tensor([item["text_len"] for item in batch], dtype=torch.long)
    fbank_lens = torch.tensor([item["fbank_len"] for item in batch], dtype=torch.long)

    padded_decoder_inputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=0)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)       # [B, T_text]
    padded_fbanks = pad_sequence(fbanks, batch_first=True, padding_value=0.0)   # [B, T_audio, 80]

    speech_mask=calculate_mask(fbank_lens, padded_fbanks.size(1))    # [B, T]
    text_mask= calculate_mask(text_lens, padded_texts.size(1) + 1).unsqueeze(1) & causal_mask(padded_texts.size(0), padded_texts.size(1) + 1)
    text_mask = text_mask.unsqueeze(1)  
    return {
        "decoder_input": padded_decoder_inputs,
        "text": padded_texts,
        "text_mask": text_mask,
        "text_len" : text_lens,
        "fbank_len" : fbank_lens,
        "fbank": padded_fbanks,
        "fbank_mask": speech_mask
    }

# self.augment = torch.nn.Sequential(
#     torchaudio.transforms.TimeMasking(40),
#     torchaudio.transforms.FrequencyMasking(30)
# )

# def get_fbank(...):
#     fbank = self.fbank(waveform)
#     if self.training:
#         fbank = self.augment(fbank)
#     return fbank.squeeze(0)