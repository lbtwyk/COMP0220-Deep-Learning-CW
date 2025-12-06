"""
Qwen3 Finetuning Pipeline using TRL SFTTrainer with LoRA.

This script finetunes a Qwen3 model on three datasets:
1. knowledge_dataset.json - Deaf culture & ASL knowledge
2. train.json - Sign language QA pairs
3. Education Dialogue Dataset - Multi-turn teacher/student conversations

Usage:
    python train_qwen3.py                          # Default training
    python train_qwen3.py --test                   # Quick test run
    python train_qwen3.py --hardware mps           # Apple Silicon
    python train_qwen3.py --hardware low_memory    # Low VRAM GPU
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from data_preprocessing import create_unified_dataset, split_dataset, preview_dataset
from config import (
    PipelineConfig,
    get_config_for_hardware,
    get_test_config,
)


def setup_quantization(config: PipelineConfig) -> BitsAndBytesConfig | None:
    """Setup quantization configuration if enabled."""
    if config.model.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config.model.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
        )
    elif config.model.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model_and_tokenizer(config: PipelineConfig):
    """Load the base model and tokenizer."""
    print(f"\n{'='*60}")
    print(f"Loading model: {config.model.model_name_or_path}")
    print(f"{'='*60}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        use_fast=config.model.use_fast_tokenizer,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Setup quantization
    quantization_config = setup_quantization(config)
    
    # Determine torch dtype
    torch_dtype = getattr(torch, config.model.torch_dtype)
    
    # Load model
    model_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # Handle device map
    if config.model.device_map == "mps":
        # For MPS, load to CPU first then move
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = config.model.device_map
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        **model_kwargs
    )
    
    # Move to MPS if needed
    if config.model.device_map == "mps":
        if torch.backends.mps.is_available():
            model = model.to("mps")
        else:
            print("Warning: MPS not available, using CPU")
            model = model.to("cpu")
    
    # Enable gradient checkpointing
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training if quantized
    if config.model.load_in_4bit or config.model.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    print(f"Model loaded successfully!")
    print(f"  Parameters: {model.num_parameters():,}")
    
    return model, tokenizer


def setup_lora(model, config: PipelineConfig):
    """Apply LoRA configuration to the model."""
    if not config.lora.use_lora:
        print("LoRA disabled, training full model")
        return model
    
    print(f"\n{'='*60}")
    print("Setting up LoRA")
    print(f"{'='*60}")
    
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total parameters: {total_params:,}")
    
    return model


def prepare_data(config: PipelineConfig):
    """Load and prepare the training data."""
    print(f"\n{'='*60}")
    print("Preparing data")
    print(f"{'='*60}")
    
    base_dir = Path(config.data.data_dir)
    
    # Create unified dataset
    dataset = create_unified_dataset(
        base_dir,
        include_knowledge=config.data.include_knowledge,
        include_qa=config.data.include_qa,
        include_education=config.data.include_education,
        education_max_files=config.data.education_max_files,
        education_split=config.data.education_split,
    )
    
    # Split into train/validation
    splits = split_dataset(dataset, test_size=config.data.val_split_ratio)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(splits['train']):,}")
    print(f"  Validation: {len(splits['validation']):,}")
    
    # Preview a few samples
    preview_dataset(splits['train'], num_samples=2)
    
    return splits['train'], splits['validation']


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: PipelineConfig,
) -> SFTTrainer:
    """Create the SFTTrainer for finetuning."""
    print(f"\n{'='*60}")
    print("Creating trainer")
    print(f"{'='*60}")
    
    # Create SFTConfig
    sft_config = SFTConfig(
        output_dir=config.training.output_dir,
        
        # Training schedule
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        
        # Batch size
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        
        # Learning rate
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        warmup_steps=config.training.warmup_steps,
        
        # Optimizer
        optim=config.training.optim,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        
        # Sequence length
        max_length=config.training.max_seq_length,
        
        # Logging and saving
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_steps=config.training.eval_steps,
        eval_strategy=config.training.eval_strategy,
        
        # Mixed precision
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        
        # Gradient checkpointing
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.training.gradient_checkpointing else None,
        
        # Seed
        seed=config.training.seed,
        
        # Hub
        push_to_hub=config.training.push_to_hub,
        hub_model_id=config.training.hub_model_id,
        
        # Dataloader
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        
        # Other
        remove_unused_columns=config.data.remove_unused_columns,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    print(f"Trainer created successfully!")
    print(f"  Output directory: {config.training.output_dir}")
    print(f"  Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    
    return trainer


def train(config: PipelineConfig):
    """Main training function."""
    print("\n" + "=" * 60)
    print("QWEN3 FINETUNING PIPELINE")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Apply LoRA
    model = setup_lora(model, config)
    
    # Prepare data
    train_dataset, eval_dataset = prepare_data(config)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
    
    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    if config.training.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save final model
    print(f"\n{'='*60}")
    print("Saving model...")
    print(f"{'='*60}")
    
    final_path = Path(config.training.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    print(f"Model saved to: {final_path}")
    
    # Push to hub if configured
    if config.training.push_to_hub and config.training.hub_model_id:
        print(f"Pushing to hub: {config.training.hub_model_id}")
        trainer.push_to_hub()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Finetuning Pipeline")
    parser.add_argument(
        "--hardware",
        type=str,
        default="default",
        choices=["default", "low_memory", "high_memory", "a800", "mps", "cpu"],
        help="Hardware preset configuration"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test with minimal configuration"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name/path"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Override max sequence length"
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA (full finetuning)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    if args.test:
        config = get_test_config()
        print("Using TEST configuration (minimal training)")
    else:
        config = get_config_for_hardware(args.hardware)
        print(f"Using {args.hardware.upper()} configuration")
    
    # Apply overrides
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.model:
        config.model.model_name_or_path = args.model
    if args.epochs:
        config.training.num_train_epochs = args.epochs
        config.training.max_steps = -1
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.max_seq_length:
        config.training.max_seq_length = args.max_seq_length
    if args.no_lora:
        config.lora.use_lora = False
    if args.resume:
        config.training.resume_from_checkpoint = args.resume
    
    # Set data directory to script location
    config.data.data_dir = str(Path(__file__).parent)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
