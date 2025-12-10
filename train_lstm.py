"""
LSTM Training Script.

Train a sequence-to-sequence LSTM model from scratch for comparison
with the finetuned Qwen3 model.

Usage:
    python train_lstm.py                    # Default training
    python train_lstm.py --epochs 100       # More epochs
    python train_lstm.py --use_education    # Include education dialogue data
    python train_lstm.py --test             # Quick test run
"""

import argparse
import time
from pathlib import Path
from typing import Optional
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from lstm_model import LSTMConfig, Seq2SeqLSTM, create_model
from lstm_data import prepare_data, Vocabulary


def train_epoch(
    model: Seq2SeqLSTM,
    train_loader,
    optimizer,
    criterion,
    config: LSTMConfig,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    # Decay teacher forcing over epochs
    teacher_forcing_ratio = max(0.2, config.teacher_forcing_ratio * (0.95 ** epoch))
    
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch in progress:
        src = batch["src"].to(config.device)
        trg = batch["trg"].to(config.device)
        src_len = batch["src_len"]
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, src_len, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # Compute loss (ignore padding and first token)
        # output: [batch, trg_len, vocab_size]
        # trg: [batch, trg_len]
        output = output[:, 1:].contiguous().view(-1, config.vocab_size)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(train_loader)


def evaluate(
    model: Seq2SeqLSTM,
    val_loader,
    criterion,
    config: LSTMConfig
) -> float:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            src = batch["src"].to(config.device)
            trg = batch["trg"].to(config.device)
            src_len = batch["src_len"]
            
            # Forward pass (no teacher forcing during eval)
            output = model(src, src_len, trg, teacher_forcing_ratio=0)
            
            output = output[:, 1:].contiguous().view(-1, config.vocab_size)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def generate_samples(
    model: Seq2SeqLSTM,
    val_loader,
    vocab: Vocabulary,
    config: LSTMConfig,
    num_samples: int = 3
) -> list:
    """Generate sample outputs for qualitative evaluation."""
    model.eval()
    samples = []
    
    with torch.no_grad():
        batch = next(iter(val_loader))
        src = batch["src"].to(config.device)
        src_len = batch["src_len"]
        trg = batch["trg"]
        
        # Generate
        generated, _ = model.generate(
            src[:num_samples],
            src_len[:num_samples],
            max_length=config.max_output_length,
            temperature=0.7
        )
        
        for i in range(min(num_samples, len(src))):
            question = vocab.decode(src[i].tolist())
            reference = vocab.decode(trg[i].tolist())
            prediction = vocab.decode(generated[i].tolist())
            
            samples.append({
                "question": question,
                "reference": reference,
                "prediction": prediction
            })
    
    return samples


def save_checkpoint(
    model: Seq2SeqLSTM,
    optimizer,
    vocab: Vocabulary,
    config: LSTMConfig,
    epoch: int,
    train_loss: float,
    val_loss: float,
    output_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": {
            "vocab_size": config.vocab_size,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "bidirectional_encoder": config.bidirectional_encoder,
            "use_attention": config.use_attention,
            "max_input_length": config.max_input_length,
            "max_output_length": config.max_output_length,
        }
    }
    
    # Save latest
    torch.save(checkpoint, output_dir / "checkpoint_latest.pt")
    
    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / "checkpoint_best.pt")
    
    # Save vocabulary
    vocab.save(output_dir / "vocab.json")
    
    # Save config as JSON for easy inspection
    with open(output_dir / "config.json", "w") as f:
        json.dump(checkpoint["config"], f, indent=2)


def load_checkpoint(
    checkpoint_path: Path,
    config: LSTMConfig,
    device: str
) -> tuple:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Update config from checkpoint
    for key, value in checkpoint["config"].items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    return model, checkpoint


def train(
    config: LSTMConfig,
    output_dir: Path,
    use_knowledge: bool = True,
    use_education: bool = False,
    resume_from: Optional[Path] = None
):
    """Main training function."""
    print("\n" + "=" * 60)
    print("LSTM BASELINE TRAINING")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Output directory: {output_dir}")
    
    # Prepare data
    base_dir = Path(__file__).parent
    train_loader, val_loader, vocab = prepare_data(
        base_dir, config,
        use_knowledge=use_knowledge,
        use_education=use_education
    )
    
    # Update config with actual vocab size
    config.vocab_size = len(vocab)
    print(f"Final vocabulary size: {config.vocab_size}")
    
    # Create model
    model = create_model(config)
    model.to(config.device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Loss function (ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and resume_from.exists():
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "samples": []
    }
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    for epoch in range(start_epoch, config.num_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config, epoch)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, config)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Check if best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, vocab, config, epoch,
            train_loss, val_loss, output_dir, is_best
        )
        
        # Generate samples periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            samples = generate_samples(model, val_loader, vocab, config)
            history["samples"].append({"epoch": epoch + 1, "samples": samples})
            
            print(f"\nSample predictions (epoch {epoch + 1}):")
            for s in samples[:2]:
                print(f"  Q: {s['question'][:80]}...")
                print(f"  R: {s['reference'][:80]}...")
                print(f"  P: {s['prediction'][:80]}...")
                print()
        
        # Log progress
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Best: {best_val_loss:.4f} | "
              f"Time: {elapsed:.1f}s")
    
    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")
    
    return model, vocab, history


def main():
    parser = argparse.ArgumentParser(description="Train LSTM baseline model")
    parser.add_argument("--output_dir", type=str, default="./lstm_baseline",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="LSTM hidden dimension")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Number of LSTM layers")
    parser.add_argument("--use_education", action="store_true",
                        help="Include education dialogue dataset")
    parser.add_argument("--no_knowledge", action="store_true",
                        help="Exclude knowledge dataset")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--test", action="store_true",
                        help="Quick test run with minimal config")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "mps", "cpu"],
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Create config
    config = LSTMConfig()
    
    # Apply test config
    if args.test:
        config.num_epochs = 5
        config.batch_size = 8
        config.hidden_dim = 128
        config.embedding_dim = 64
        print("Using TEST configuration")
    
    # Apply overrides
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.embedding_dim:
        config.embedding_dim = args.embedding_dim
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.device:
        config.device = args.device
    
    output_dir = Path(args.output_dir)
    resume_from = Path(args.resume) if args.resume else None
    
    # Train
    train(
        config=config,
        output_dir=output_dir,
        use_knowledge=not args.no_knowledge,
        use_education=args.use_education,
        resume_from=resume_from
    )


if __name__ == "__main__":
    main()
