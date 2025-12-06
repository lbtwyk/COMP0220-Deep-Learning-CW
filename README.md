# Qwen3 Finetuning Pipeline

A complete finetuning pipeline for Qwen3 models using LoRA (Low-Rank Adaptation) with the TRL library's SFTTrainer.

## Overview

This pipeline finetunes **Qwen/Qwen3-4B-Instruct-2507** on three data sources:
1. **knowledge_dataset/** - Directory of Deaf culture & ASL Q&A JSON files (supports both `input`/`output` and `question`/`answer` formats)
2. **knowledge_dataset/train.json** - Sign language QA pairs (~85 entries)
3. **Education-Dialogue-Dataset** - Multi-turn teacher/student conversations (40,000+ training examples)

## Project Structure

```
DL_CW/
├── train_qwen3.py          # Main training script
├── config.py               # Configuration classes
├── data_preprocessing.py   # Data loading and preprocessing
├── inference.py            # Inference/evaluation script for finetuned model
├── requirements.txt        # Python dependencies
├── knowledge_dataset/      # Deaf culture & ASL knowledge Q&A JSON files
├── Education-Dialogue-Dataset-main/
    ├── conversations_train1-5.json  # Training data
    └── conversations_eval.json      # Evaluation data
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Test Run (Quick Validation)

```bash
python train_qwen3.py --test
```

This runs a minimal training loop with:
- Smaller model (Qwen3-0.6B)
- 50 training steps
- 1 education dialogue file

### 2. Full Training

```bash
# Default configuration (uses cached Qwen/Qwen3-4B-Instruct-2507, requires ~24GB VRAM)
python train_qwen3.py

# For Apple Silicon (M1/M2/M3)
python train_qwen3.py --hardware mps

# For low VRAM GPUs (<16GB)
python train_qwen3.py --hardware low_memory

# For high VRAM GPUs (>40GB)
python train_qwen3.py --hardware high_memory
```

### 3. Custom Training

```bash
python train_qwen3.py \
    --model Qwen/Qwen3-1.7B \
    --epochs 5 \
    --lr 1e-4 \
    --batch_size 4 \
    --output_dir ./my_model
```

## Configuration Options

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--hardware` | Hardware preset (default, low_memory, high_memory, mps, cpu) | default |
| `--test` | Quick test run | False |
| `--model` | Model name/path | Qwen/Qwen3-4B |
| `--epochs` | Number of epochs | 3 |
| `--lr` | Learning rate | 2e-4 |
| `--batch_size` | Per-device batch size | 2 |
| `--max_seq_length` | Maximum sequence length | 2048 |
| `--no_lora` | Disable LoRA (full finetuning) | False |
| `--resume` | Resume from checkpoint | None |
| `--output_dir` | Output directory | ./qwen3_finetuned |

### Hardware Presets

| Preset | VRAM | Batch Size | Seq Length | Quantization |
|--------|------|------------|------------|--------------|
| default | 24GB+ | 2 | 2048 | None |
| low_memory | <16GB | 1 | 1024 | 4-bit |
| high_memory | 40GB+ | 4 | 4096 | None |
| mps | Apple Silicon | 1 | 1024 | None |
| cpu | CPU only | 1 | 512 | None |

### LoRA Configuration

Default LoRA settings (applied on top of the instruct checkpoint):
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Inference

After training, use the inference script:

```bash
# Interactive chat
python inference.py --model_path ./qwen3_finetuned/final --interactive
```

### 2. Single Prompt

```bash
python inference.py --model_path ./qwen3_finetuned/final --prompt "What is ASL?"
```

### 3. Quick Demo (built-in sample questions)

```bash
python inference.py --model_path ./qwen3_finetuned/final
```

### 4. Dataset Evaluation

By default, if you do **not** pass `--dataset_path`, evaluation will automatically use **all JSON files** under the `knowledge_dataset/` directory (e.g. `knowledge_dataset.json`, `knowledge2.json`, `knowledge3.json`, `knowledge4.json`, ...):

```bash
python inference.py --model_path ./qwen3_finetuned/final --num_samples 50
```

You can also specify a custom file or directory:

```bash
# Evaluate on a specific JSON file (knowledge-style or QA-style)
python inference.py \
    --model_path ./qwen3_finetuned/final \
    --dataset_path ./knowledge_dataset/knowledge_dataset.json \
    --num_samples 50

# Evaluate on a different dataset directory
python inference.py \
    --model_path ./qwen3_finetuned/final \
    --dataset_path ./knowledge_dataset \
    --num_samples 100
```

Supported item formats for evaluation are the same as for training:

- `{ "input": "...", "output": "..." }`
- `{ "question": "...", "answer": "..." }`
- `{ "context": "...", "question": "...", "answer": "..." }`

All of them are mapped to the unified conversational format with the same system prompt used in training.

## Dataset Format

The pipeline converts all datasets to a unified conversational format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a Deaf culture specialist and ASL tutor. Answer clearly."},
    {"role": "user", "content": "What is ASL?"},
    {"role": "assistant", "content": "ASL stands for American Sign Language..."}
  ]
}
```

### Dataset Sources

All datasets are normalized to the above conversational format with a *unified system prompt*.

- **knowledge_dataset/**
  - Multiple JSON files (e.g. `knowledge_dataset.json`, `knowledge2.json`, `knowledge3.json`, `knowledge4.json`, ...)
  - Supported formats per item:
    - Knowledge Q&A: `{ "input": "...", "output": "..." }` (optional `"system"` is ignored)
    - QA: `{ "question": "...", "answer": "..." }` (optional `"context"` is ignored)
- **knowledge_dataset/train.json**
  - `{ "context": "...", "question": "...", "answer": "..." }`
  - Only `question` and `answer` are used during training/eval
- **Education-Dialogue-Dataset-main/**
  - Teacher/Student multi-turn dialogues converted into `messages` with a topic-specific system message

## Evaluation

The same `inference.py` script can run interactive chat, single-prompt inference, or dataset evaluation.

### 1. Interactive Chat

## Training Tips

1. **Start small**: Use `--test` to validate the pipeline before full training
2. **Monitor memory**: Watch GPU memory usage and adjust batch size accordingly
3. **Use gradient accumulation**: Increase effective batch size without more memory
4. **Enable gradient checkpointing**: Saves memory at the cost of speed
5. **Use 4-bit quantization**: For limited VRAM, use `--hardware low_memory`

## Monitoring Training

Training logs are saved to TensorBoard:

```bash
tensorboard --logdir ./qwen3_finetuned
```

## Model Outputs

After training, you'll find:
- `./qwen3_finetuned/checkpoint-*/` - Intermediate checkpoints
- `./qwen3_finetuned/final/` - Final model with tokenizer
- `./qwen3_finetuned/runs/` - TensorBoard logs

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size`
- Increase `gradient_accumulation_steps` in config
- Use `--hardware low_memory` for 4-bit quantization
- Reduce `--max_seq_length`

### Slow Training on MPS
- MPS is slower than CUDA; this is expected
- Use `--hardware mps` preset for optimized settings

### Model Not Learning
- Check learning rate (try 1e-4 to 3e-4)
- Increase epochs
- Verify data is loading correctly with `python data_preprocessing.py`

## License

This project is for educational purposes. Please check the licenses of:
- Qwen3 model: Apache 2.0
- TRL library: Apache 2.0
- PEFT library: Apache 2.0
- Datasets: Check individual dataset licenses
