import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import numpy as np
import librosa
from glob import glob
import os

# [{idx : {encoded_text : Tensor, wav_path : text} }]


def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    import json

    with open(path, "r", encoding='utf-8') as f:
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

def compute_gmvn(voice_path, sample_rate=16000):
    wav_files = glob(os.path.join(voice_path, "**", "*.wav"), recursive=True)

    win_length = int(0.025 * sample_rate)   # 25ms = 400 samples
    hop_length = int(0.010 * sample_rate)   # 10ms = 160 samples
    
    sum_feats = torch.zeros(192)
    sum_squares = torch.zeros(192)
    total_frames = 0

    for file in tqdm(wav_files):  # dataset là list/tập của waveform tensors
        with torch.no_grad():
            waveform, _ = librosa.load(file, sr=sample_rate)
            stft = librosa.stft(waveform, n_fft=512, win_length=win_length, hop_length=hop_length)
            mag = np.abs(stft[:64, :])  # Lấy 64 bins đầu tiên (low frequencies)
            log_mag = np.log1p(mag)  # log(1 + x)
            log_mag = log_mag.T  # [T, 64]

            stacked_feats = []
            for i in range(len(log_mag) - 6):  # skip rate = 3
                if i % 3 == 0:
                    stacked = np.concatenate([log_mag[i], log_mag[i+3], log_mag[i+6]])  # [192]
                    stacked_feats.append(stacked)

            stacked_feats = torch.tensor(np.array(stacked_feats), dtype=torch.float32)  # [T', 192]

            total_frames += stacked_feats.shape[0]
            sum_feats += stacked_feats.sum(dim=0)
            sum_squares += (stacked_feats ** 2).sum(dim=0)

    mean = sum_feats / total_frames
    std = (sum_squares / total_frames - mean**2).sqrt()
    return mean, std

def stack_context(x, left=3, right=1):
    """x: (T, D) -> (T, (left+1+right)*D) | pad biên bằng replicate."""
    T, D = x.shape
    pads = []
    for off in range(-left, right + 1):
        idx = np.clip(np.arange(T) + off, 0, T - 1)
        pads.append(x[idx])
    return np.concatenate(pads, axis=1)

def subsample(x, base_hop_ms=10, target_hop_ms=30):
    stride = target_hop_ms // base_hop_ms
    return x[::stride]

class Speech2Text(Dataset):
    def __init__(self, json_path, vocab_path, gmvn_mean = None, gmvn_std = None):
        super().__init__()
        self.data = load_json(json_path)
        self.vocab = Vocab(vocab_path)
        self.sos_token = self.vocab.get_sos_token()
        self.eos_token = self.vocab.get_eos_token()
        self.pad_token = self.vocab.get_pad_token()
        self.unk_token = self.vocab.get_unk_token()
        
        self.gmvn_mean = gmvn_mean
        self.gmvn_std = gmvn_std
        # stats = torch.load(cmvn_stats) 
        # self.cmvn_mean = stats['mean']
        # self.cmvn_std = stats['std']
            
    def __len__(self):
        return len(self.data)

    def extract_features(self, wav_file, sr=16000):
        y, sr = librosa.load(wav_file, sr=sr)
        win_length = int(0.025 * sr)   # 25 ms
        hop_length = int(0.010 * sr)   # 10 ms
        # n_fft = next power of 2 >= win_length
        n_fft = 1
        while n_fft < win_length:
            n_fft *= 2

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=40, n_fft=n_fft,
            win_length=win_length, hop_length=hop_length,
            window='hann', power=2.0, center=True
        )
        # log-mel (dB)
        x = librosa.power_to_db(S, ref=np.max).T   # (T, 40)
        
        mu = x.mean(axis=0, keepdims=True)
        sg = x.std(axis=0, keepdims=True) + 1e-8
        x = (x - mu) / sg
        x = stack_context(x, left=3, right=1) 
        return torch.tensor(subsample(x, 10, 30))


    def __getitem__(self, idx):
        current_item = self.data[idx]
        wav_path = current_item["wav_path"]
        encoded_text = torch.tensor(current_item["encoded_text"] + [self.eos_token], dtype=torch.long)
        decoder_input = torch.tensor([self.sos_token] + current_item["encoded_text"] + [self.pad_token], dtype=torch.long)
        fbank = self.extract_features(wav_path)  # [T, 80]
        
        return {
            "text": encoded_text,        # [T_text]
            "fbank": fbank,              # [T_audio, 80]
            "text_len": len(encoded_text),
            "fbank_len": fbank.shape[0],
            "decoder_input": decoder_input,  # [T_text + 1]
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