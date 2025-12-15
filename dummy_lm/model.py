"""
Dummy Transformer Language Model for Educational Content
A from-scratch PyTorch implementation for COMP0220 demonstration
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer decoder block with self-attention"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attn_mask: Causal attention mask
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=True)
        x = self.ln1(x + self.dropout(attn_out))
        
        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x


class DummyTransformerLM(nn.Module):
    """
    Decoder-only Transformer Language Model
    
    A simplified GPT-style model for educational text generation
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying between input and output embeddings
        self.head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Model info
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'max_len': max_len,
            'dropout': dropout
        }
    
    def _init_weights(self, module):
        """Initialize weights with small values"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            labels: Target token IDs for loss computation
            
        Returns:
            logits or (loss, logits) if labels provided
        """
        # Get embeddings
        x = self.tok_emb(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Cross-entropy loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fn(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if loss is not None:
            return loss, logits
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively
        
        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get logits for last position
            with torch.no_grad():
                logits = self(input_ids)[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    model = DummyTransformerLM(
        vocab_size=10000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Config: {model.config}")
    
    # Test forward pass
    x = torch.randint(0, 1000, (2, 128))
    loss, logits = model(x, labels=x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
