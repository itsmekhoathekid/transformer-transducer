import torch
import torch.nn.functional as F
from utils.dataset import Speech2Text, speech_collate_fn
from models.model import TransformerTransducer
import yaml
import os
import argparse
from tqdm import tqdm
from jiwer import wer, cer

ENC_OUT_KEY = "encoder_out"
SPEECH_IDX_KEY = "speech_idx"
HIDDEN_STATE_KEY = "hidden_state"
DECODER_OUT_KEY = "decoder_out"
PREDS_KEY = "preds"
PREV_HIDDEN_STATE_KEY = "prev_hidden_state"

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(config: dict, vocab_len: int, device: torch.device, epoch) -> TransformerTransducer:
    checkpoint_path = os.path.join(
        config['training']['save_path'],
        f"transformer_transducer_epoch_{epoch}"
    )
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = TransformerTransducer(
        in_features=config['model']['in_features'],
        n_classes=vocab_len,
        n_enc_layers=config['model']['n_enc_layers'],
        n_dec_layers=config['model']['n_dec_layers'],
        d_model=config['model']['d_model'],
        ff_size=config['model']['ff_size'],
        h=config['model']['h'],
        joint_size=config['model']['joint_size'],
        p_dropout=config['model']['p_dropout']
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

class TransducerPredictor:
    def __init__(self, model, vocab, device, sos=1, eos=2, blank=4):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.sos = sos
        self.eos = eos
        self.blank = blank
        self.idx2token = {idx: token for token, idx in vocab.items()}

    def greedy_decode(self, encoder_outputs: torch.Tensor, max_length: int = 100, typee : str = "normal") -> str:
        """
        Greedy decode for Transformer Transducer (clean version, no blank blocking).
        Args:
            encoder_outputs (Tensor): (B, T, D)
            max_length (int): max decoding time steps

        Returns:
            str: predicted transcription
        """
        B, T, _ = encoder_outputs.size()
        assert B == 1, "This greedy_decode only supports batch size = 1."
        pred_tokens = []

        targets = encoder_outputs.new_tensor([[self.sos]], dtype=torch.long).to(self.device)
        
        
        for t in range(min(T, max_length)):
            text_mask = torch.ones_like(targets, dtype=torch.bool, device=self.device)
            
            decoder_output, _ = self.model.decoder(targets, text_mask)               # (B, U, D)
            dec_proj = decoder_output[:, -1, :]            # (B, 1, D)
            enc_step = encoder_outputs[:, t, :]                                     # (B, 1, D)

            logits = self.model._join(enc_step, dec_proj)                       # (B, 1, 1, V)
            logits = F.softmax(logits.squeeze(1).squeeze(1), dim=-1)   

            top_token = logits.argmax(dim=-1)                                   # (B,)
            token_id = top_token.item()

            
            if token_id == self.eos:
                break
            
            if token_id != self.blank and token_id != self.eos:
                pred_tokens.append(token_id)
                targets = torch.cat([targets, top_token.unsqueeze(1)], dim=1)

        tokens = [self.idx2token.get(t, "") for t in pred_tokens]
        if typee == "phoneme":
            return "".join(tokens).replace("<space>", " ").strip()
        
        return " ".join(tokens)





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of the model to load")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = Speech2Text(
        json_path=config['training']['test_path'],
        vocab_path=config['training']['vocab_path'],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=speech_collate_fn
    )
    vocab = test_dataset.vocab.stoi
    vocab_len = len(vocab)

    model = load_model(config, vocab_len, device, epoch=args.epoch)
    predictor = TransducerPredictor(model, vocab, device, sos=1, eos=2, blank=0)

    all_predictions = []
    all_references = []

    j = 0 
    with open(config["training"]["result"], 'w', encoding='utf-8') as fout:
        for batch in tqdm(test_loader, desc="Inference"):
            speech = batch["fbank"].to(device)
            speech_mask = batch["fbank_mask"].to(device)
            
            
    
            encoder_out, _ = model.encoder(speech, speech_mask)
            pred_transcription = predictor.greedy_decode(encoder_out, max_length=encoder_out.size(1), typee = config['training']['type'])
            all_predictions.append(pred_transcription)
            
            ref_ids = batch["text"].squeeze(0).tolist()
            idx2token = {idx: token for token, idx in vocab.items()}
            ref_tokens = [idx2token.get(token, "") for token in ref_ids][:-1]
            ref_transcription = " ".join(ref_tokens)
            print("🔊", pred_transcription)
            print("🎯", ref_transcription)
            all_references.append(ref_transcription)
            fout.write(f"Predict text: {pred_transcription}\n")
            fout.write(f"Ground truth: {ref_transcription}\n")
            fout.write("---------------\n")
    
   

    wer_score = wer(all_references, all_predictions)
    cer_score = cer(all_references, all_predictions)

    

    print(f"Average WER: {wer_score:.4f}")
    print(f"Average CER: {cer_score:.4f}")

if __name__ == "__main__":
    main()
