# Transformer Transducer — My Implementation

> Implementation of **Transformer Transducer** for end-to-end speech recognition, inspired by the paper:  
> **Self-Attention Transducers for End-to-End Speech Recognition** (2019).  
> 📄 Paper: https://www.isca-archive.org/interspeech_2019/tian19b_interspeech.pdf

## 🔗 Repository
- GitHub: https://github.com/itsmekhoathekid/transformer-transducer

## 🚀 Quickstart

### 1) Clone & Setup
```
bash
git clone https://github.com/itsmekhoathekid/transformer-transducer
cd transformer-transducer
```

### 2) Download & Prepare Dataset
This will download the datasets configured inside the script and generate manifests/features as needed.
```
bash
bash ./prep_data.sh
```

### 3) Train
Train with a YAML/JSON config of your choice.
```
bash
python train.py --config path/to/train_config.yaml
```

### 4) Inference (example)
```
bash
python infererence.py --config path/to/train_config.yaml --epoch num_epoch
```

## 📦 Project Layout (typical)
```
transformer-transducer/
├── prep_data.sh                 # dataset download & preprocessing
├── train.py                     # training entry point
├── inference.py                     # inference script (optional)
├── configs/                     # training configs (yaml/json)
├── models/                    # model, losses, data, utils
│   ├── model.py
│   ├── encoder.py
│   ├── decoder.py
│   └── ...
├── utils/ 
└── README.md
```

