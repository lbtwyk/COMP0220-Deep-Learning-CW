"""
Training Script for Dummy Transformer LM
Trains on the knowledge_dataset.json for educational QA
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import os
from tqdm import tqdm
from collections import Counter
import argparse

from model import DummyTransformerLM


class SimpleTokenizer:
    """Character-level tokenizer with special tokens"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<sep>': 4  # Separator between input and output
        }
        self.vocab_size = 0
    
    def build_vocab(self, texts: list):
        """Build vocabulary from list of texts"""
        # Start with special tokens
        self.char_to_idx = dict(self.special_tokens)
        idx = len(self.special_tokens)
        
        # Count all characters
        all_chars = set()
        for text in texts:
            all_chars.update(set(text))
        
        # Add characters to vocabulary
        for char in sorted(all_chars):
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                idx += 1
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
        return self
    
    def encode(self, text: str) -> list:
        """Encode text to token IDs"""
        return [self.char_to_idx.get(c, self.special_tokens['<unk>']) for c in text]
    
    def decode(self, ids: list) -> str:
        """Decode token IDs to text"""
        return ''.join([self.idx_to_char.get(i, '?') for i in ids 
                       if i not in [0, 2, 3]])  # Skip special tokens
    
    def save(self, path: str):
        """Save tokenizer vocabulary"""
        with open(path, 'w') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'special_tokens': self.special_tokens
            }, f)
    
    def load(self, path: str):
        """Load tokenizer vocabulary"""
        with open(path, 'r') as f:
            data = json.load(f)
            self.char_to_idx = data['char_to_idx']
            self.special_tokens = data['special_tokens']
            self.idx_to_char = {int(v): k for k, v in self.char_to_idx.items()}
            self.vocab_size = len(self.char_to_idx)
        return self


class QADataset(Dataset):
    """Dataset for Question-Answer pairs"""
    
    def __init__(self, data: list, tokenizer: SimpleTokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        for item in data:
            # Format: <bos> question <sep> answer <eos>
            question = item['input']
            answer = item['output']
            
            # Encode
            bos = [tokenizer.special_tokens['<bos>']]
            sep = [tokenizer.special_tokens['<sep>']]
            eos = [tokenizer.special_tokens['<eos>']]
            
            q_tokens = tokenizer.encode(question)
            a_tokens = tokenizer.encode(answer)
            
            # Combine: bos + question + sep + answer + eos
            tokens = bos + q_tokens + sep + a_tokens + eos
            
            # Truncate if needed
            if len(tokens) > max_len:
                tokens = tokens[:max_len-1] + eos
            
            self.samples.append(tokens)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)


def collate_fn(batch):
    """Pad sequences to same length"""
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    
    return padded


def train(args):
    """Main training function"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # Build tokenizer
    print("Building vocabulary...")
    tokenizer = SimpleTokenizer()
    all_texts = [item['input'] + item['output'] for item in data]
    tokenizer.build_vocab(all_texts)
    
    # Create dataset and dataloader
    dataset = QADataset(data, tokenizer, max_len=args.max_len)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = DummyTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        pad_token_id=0
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            
            # Forward
            loss, _ = model(batch, labels=batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': model.config
            }, save_path)
            print(f"✓ Saved best model (loss: {avg_loss:.4f})")
    
    # Save final model and tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': model.config
    }, final_path)
    
    tokenizer.save(os.path.join(args.output_dir, 'tokenizer.json'))
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Final model saved to: {final_path}")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 60)
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train Dummy Transformer LM')
    
    # Data
    parser.add_argument('--data_path', type=str, 
                       default='../knowledge_dataset.json',
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, 
                       default='./checkpoints',
                       help='Output directory for model')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=512,
                       help='Feed-forward dimension')
    parser.add_argument('--max_len', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
