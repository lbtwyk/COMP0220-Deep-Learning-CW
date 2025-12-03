"""
Configuration for Qwen3 finetuning pipeline.
Contains model, training, and LoRA hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    # Default to the cached instruct checkpoint for this project
    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    
    # Alternative smaller models for testing
    # model_name_or_path: str = "Qwen/Qwen3-1.7B-Instruct"
    # model_name_or_path: str = "Qwen/Qwen3-0.6B"
    
    # Tokenizer settings
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = True
    
    # Model loading settings
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    device_map: str = "auto"
    
    # Quantization (for memory efficiency)
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    # Enable LoRA
    use_lora: bool = True
    
    # LoRA hyperparameters
    r: int = 16  # Rank of the low-rank matrices
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.05
    
    # Target modules for Qwen3
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ])
    
    # Bias handling
    bias: str = "none"  # "none", "all", "lora_only"
    
    # Task type
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Output directory
    output_dir: str = "./qwen3_finetuned"
    
    # Training hyperparameters
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use num_train_epochs
    
    # Batch size settings
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch size = 2 * 8 = 16
    
    # Learning rate settings
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio
    
    # Optimizer
    optim: str = "adamw_torch"  # "adamw_torch", "adamw_8bit", "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Sequence length
    max_seq_length: int = 2048
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "steps", "epoch", "no"
    
    # Mixed precision
    bf16: bool = True
    fp16: bool = False
    
    # Gradient checkpointing (saves memory)
    gradient_checkpointing: bool = True
    
    # Seed for reproducibility
    seed: int = 42
    
    # Hub settings (optional)
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None
    
    # Dataloader settings
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True


@dataclass
class DataConfig:
    """Data configuration."""
    # Base directory containing datasets
    data_dir: str = "."
    
    # Dataset inclusion flags
    include_knowledge: bool = True
    include_qa: bool = True
    include_education: bool = True
    
    # Education dialogue settings
    education_max_files: Optional[int] = None  # None = use all files
    education_split: str = "train"
    
    # Train/validation split
    val_split_ratio: float = 0.05
    
    # Preprocessing
    remove_unused_columns: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        # Ensure output directory exists
        Path(self.training.output_dir).mkdir(parents=True, exist_ok=True)


# Preset configurations for different hardware
def get_config_for_hardware(hardware: str = "default") -> PipelineConfig:
    """
    Get configuration preset for different hardware setups.
    
    Args:
        hardware: One of "default", "low_memory", "high_memory", "mps"
    """
    config = PipelineConfig()
    
    if hardware == "low_memory":
        # For GPUs with < 16GB VRAM
        config.model.load_in_4bit = True
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 16
        config.training.max_seq_length = 1024
        config.lora.r = 8
        config.lora.lora_alpha = 16
        
    elif hardware == "high_memory":
        # For GPUs with > 40GB VRAM
        config.training.per_device_train_batch_size = 4
        config.training.gradient_accumulation_steps = 4
        config.training.max_seq_length = 4096
        config.lora.r = 32
        config.lora.lora_alpha = 64
        
    elif hardware == "mps":
        # For Apple Silicon (M1/M2/M3)
        config.model.torch_dtype = "float16"
        config.model.device_map = "mps"
        config.model.load_in_4bit = False  # BnB not fully supported on MPS
        config.model.load_in_8bit = False
        config.training.bf16 = False
        config.training.fp16 = True
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 16
        config.training.max_seq_length = 1024
        config.training.dataloader_num_workers = 0  # MPS works better with 0
        
    elif hardware == "cpu":
        # For CPU-only training (very slow, for testing only)
        config.model.torch_dtype = "float32"
        config.model.device_map = "cpu"
        config.model.load_in_4bit = False
        config.model.load_in_8bit = False
        config.training.bf16 = False
        config.training.fp16 = False
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 8
        config.training.max_seq_length = 512
        config.training.gradient_checkpointing = False
    
    return config


# Quick test configuration
def get_test_config() -> PipelineConfig:
    """Get a minimal configuration for quick testing."""
    config = PipelineConfig()
    config.model.model_name_or_path = "Qwen/Qwen3-0.6B"  # Smaller model
    config.training.max_steps = 50
    config.training.logging_steps = 5
    config.training.save_steps = 25
    config.training.eval_steps = 25
    config.data.education_max_files = 1
    config.training.per_device_train_batch_size = 1
    config.training.gradient_accumulation_steps = 4
    return config


if __name__ == "__main__":
    # Print default configuration
    config = PipelineConfig()
    print("Default Configuration:")
    print(f"  Model: {config.model.model_name_or_path}")
    print(f"  LoRA rank: {config.lora.r}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Batch size: {config.training.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
