# Transformers-PytorchLightning

Hey there! 👋 This is my implementation of the Transformer architecture from scratch using PyTorch Lightning. The main goal of this project was to deeply understand how Transformers work by implementing one myself, rather than using existing implementations like Hugging Face's transformers.

## Why This Project?

While there are plenty of production-ready Transformer implementations out there, building one from scratch is an incredible learning experience. This project helped me:
- Understand the internal mechanics of Transformers
- Get hands-on experience with PyTorch Lightning's organized training approach
- Learn how to implement complex architectures in a clean, maintainable way

## What's Inside?

I've built a basic but functional Transformer that includes all the key components:
- Multi-head self-attention mechanism
- Position-wise feed-forward networks
- Encoder and decoder stacks
- All the essential bells and whistles (positional encoding, layer normalization, etc.)

The implementation is intentionally straightforward and well-documented to serve as a learning resource.

## Getting Started

1. Clone this repo:
```bash
git clone https://github.com/yourusername/Transformers-PytorchLightning.git
cd Transformers-PytorchLightning
```

2. Install what you need:

```bash 
pip install -r requirements.txt
```

3. Run the training:

```bash 
python src/train.py
```

## Current Configuration
I've kept the model relatively small for faster training and experimentation:

* Embedding dimension: 128
* Attention heads: 4
* Encoder/decoder layers: 3
* Feed-forward dimension: 512
* Dropout: 0.1
* Learning rate: 0.0001

Training setup:

* Batch size: 32
* Epochs: 50
* Training samples: 1000
* Validation samples: 200
* Test samples: 100

All these parameters can be easily tweaked in `train.py` if you want to experiment!
