# Dummy Transformer Language Model

A from-scratch PyTorch implementation of a decoder-only Transformer for educational text generation.

## Architecture

- **Model Type**: Decoder-only Transformer (GPT-style)
- **Tokenizer**: Character-level
- **Default Config**:
  - d_model: 256
  - n_heads: 4
  - n_layers: 4
  - d_ff: 512
  - Parameters: ~3M

## Files

| File | Description |
|------|-------------|
| `model.py` | Transformer model implementation |
| `train.py` | Training script with tokenizer and data loading |
| `inference.py` | Interactive inference script |

## Usage

### Training

```bash
cd dummy_lm
python train.py --data_path ../knowledge_dataset.json --epochs 20
```

### Training Options

```
--data_path     Path to JSON dataset (default: ../knowledge_dataset.json)
--output_dir    Output directory (default: ./checkpoints)
--d_model       Model dimension (default: 256)
--n_heads       Attention heads (default: 4)
--n_layers      Transformer layers (default: 4)
--epochs        Training epochs (default: 20)
--batch_size    Batch size (default: 16)
--lr            Learning rate (default: 3e-4)
```

### Inference

```bash
python inference.py --checkpoint ./checkpoints/best_model.pt
```

## Dataset Format

Expects JSON with system/input/output format:
```json
[
  {
    "system": "...",
    "input": "Question about ASL?",
    "output": "Answer about ASL."
  }
]
```

## Model Components

1. **PositionalEncoding**: Sinusoidal position embeddings
2. **TransformerBlock**: Self-attention + FFN with layer norms
3. **DummyTransformerLM**: Full model with embedding, blocks, and output head

## Limitations

- Small model size (~3M params) limits generalization
- Character-level tokenization is inefficient
- Trained on limited dataset (~500 samples)

For production use, see the fine-tuned Qwen3-4B model in `/qwen3_finetuning`.
