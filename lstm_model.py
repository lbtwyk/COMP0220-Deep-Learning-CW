"""
LSTM Sequence-to-Sequence Model for Question Answering.

A baseline model trained from scratch to compare against finetuned Qwen3.
Uses encoder-decoder architecture with attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import math


@dataclass
class LSTMConfig:
    """Configuration for LSTM seq2seq model."""
    # Vocabulary
    vocab_size: int = 10000
    pad_token_id: int = 0
    sos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    # Model architecture
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional_encoder: bool = True
    use_attention: bool = True
    
    # Sequence lengths
    max_input_length: int = 128
    max_output_length: int = 256
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50
    gradient_clip: float = 1.0
    teacher_forcing_ratio: float = 0.5
    
    # Data
    min_word_freq: int = 2  # Minimum frequency to include word in vocab
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Encoder(nn.Module):
    """Bidirectional LSTM encoder."""
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional_encoder
        
        self.embedding = nn.Embedding(
            config.vocab_size, 
            config.embedding_dim, 
            padding_idx=config.pad_token_id
        )
        
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional_encoder,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Project bidirectional hidden states to decoder size
        if config.bidirectional_encoder:
            self.fc_hidden = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            self.fc_cell = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: [batch_size, src_len] input token ids
            src_lengths: [batch_size] actual lengths (for packing)
        
        Returns:
            outputs: [batch_size, src_len, hidden_dim * num_directions]
            (hidden, cell): final states for decoder init
        """
        # Embed and dropout
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]
        
        # Pack if lengths provided
        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, (hidden, cell) = self.lstm(packed)
            # Force padding to match the input sequence length to keep masks aligned
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=src.size(1)
            )
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional hidden states
        if self.bidirectional:
            # hidden: [num_layers * 2, batch, hidden_dim]
            # Reshape to [num_layers, batch, hidden_dim * 2]
            batch_size = hidden.size(1)
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  # [num_layers, batch, hidden*2]
            hidden = torch.tanh(self.fc_hidden(hidden))  # [num_layers, batch, hidden]
            
            cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
            cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
            cell = torch.tanh(self.fc_cell(cell))
        
        return outputs, (hidden, cell)


class Attention(nn.Module):
    """Bahdanau-style additive attention."""
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        encoder_dim = config.hidden_dim * (2 if config.bidirectional_encoder else 1)
        decoder_dim = config.hidden_dim
        
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)
    
    def forward(
        self, 
        hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden: [batch_size, hidden_dim] decoder hidden state
            encoder_outputs: [batch_size, src_len, encoder_dim]
            mask: [batch_size, src_len] padding mask (1 for valid, 0 for pad)
        
        Returns:
            attention weights: [batch_size, src_len]
        """
        src_len = encoder_outputs.size(1)
        
        # Repeat hidden for each source position
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, hidden]
        
        # Concatenate and compute energy
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        
        # Mask padding positions
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """LSTM decoder with attention."""
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        
        encoder_dim = config.hidden_dim * (2 if config.bidirectional_encoder else 1)
        
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id
        )
        
        self.attention = Attention(config) if config.use_attention else None
        
        # Input: embedding + context (if attention)
        lstm_input_dim = config.embedding_dim + (encoder_dim if config.use_attention else 0)
        
        self.lstm = nn.LSTM(
            lstm_input_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        fc_input_dim = config.hidden_dim + encoder_dim + config.embedding_dim if config.use_attention else config.hidden_dim
        self.fc_out = nn.Linear(fc_input_dim, config.vocab_size)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            input_token: [batch_size] current input token
            hidden: [num_layers, batch_size, hidden_dim]
            cell: [num_layers, batch_size, hidden_dim]
            encoder_outputs: [batch_size, src_len, encoder_dim]
            mask: [batch_size, src_len]
        
        Returns:
            prediction: [batch_size, vocab_size] logits
            hidden, cell: updated states
            attention_weights: [batch_size, src_len]
        """
        input_token = input_token.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch, 1, emb_dim]
        
        attention_weights = None
        
        if self.attention is not None:
            # Use top layer hidden state for attention
            attention_weights = self.attention(hidden[-1], encoder_outputs, mask)
            
            # Compute context vector
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, enc_dim]
            
            # Concatenate embedding and context
            lstm_input = torch.cat([embedded, context], dim=2)
        else:
            lstm_input = embedded
            context = None
        
        # LSTM step
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Compute prediction
        if self.attention is not None:
            output = torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1)
        else:
            output = output.squeeze(1)
        
        prediction = self.fc_out(output)
        
        return prediction, hidden, cell, attention_weights


class Seq2SeqLSTM(nn.Module):
    """Complete sequence-to-sequence model with encoder-decoder architecture."""
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor],
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Training forward pass.
        
        Args:
            src: [batch_size, src_len] source tokens
            src_lengths: [batch_size] source lengths
            trg: [batch_size, trg_len] target tokens
            teacher_forcing_ratio: probability of using teacher forcing
        
        Returns:
            outputs: [batch_size, trg_len, vocab_size] predictions
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        
        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, self.config.vocab_size).to(src.device)
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # Create mask for attention
        mask = (src != self.config.pad_token_id).float()
        
        # First decoder input is SOS token
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            prediction, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            
            outputs[:, t] = prediction
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        
        return outputs
    
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate output sequence (inference).
        
        Args:
            src: [batch_size, src_len] source tokens
            src_lengths: [batch_size] source lengths
            max_length: maximum output length
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling threshold
        
        Returns:
            generated: [batch_size, gen_len] generated tokens
            attention_weights: list of attention weights per step
        """
        if max_length is None:
            max_length = self.config.max_output_length
        
        batch_size = src.size(0)
        device = src.device
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        mask = (src != self.config.pad_token_id).float()
        
        # Start with SOS
        input_token = torch.full((batch_size,), self.config.sos_token_id, dtype=torch.long, device=device)
        
        generated = [input_token]
        attention_weights = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            prediction, hidden, cell, attn = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            
            if attn is not None:
                attention_weights.append(attn)
            
            # Apply temperature
            logits = prediction / temperature
            
            # Top-k sampling
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            
            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sample or argmax
            if temperature > 0:
                probs = F.softmax(logits, dim=-1)
                input_token = torch.multinomial(probs, 1).squeeze(1)
            else:
                input_token = logits.argmax(1)
            
            # Replace finished sequences with padding
            input_token = input_token.masked_fill(finished, self.config.pad_token_id)
            generated.append(input_token)
            
            # Check for EOS
            finished = finished | (input_token == self.config.eos_token_id)
            if finished.all():
                break
        
        generated = torch.stack(generated, dim=1)
        return generated, attention_weights
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: LSTMConfig) -> Seq2SeqLSTM:
    """Create and initialize the model."""
    model = Seq2SeqLSTM(config)
    
    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'lstm' in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model


if __name__ == "__main__":
    # Test model creation
    config = LSTMConfig()
    model = create_model(config)
    
    print(f"Model created with {model.count_parameters():,} parameters")
    print(f"Config: embedding_dim={config.embedding_dim}, hidden_dim={config.hidden_dim}")
    print(f"Device: {config.device}")
    
    # Test forward pass
    batch_size = 4
    src_len = 20
    trg_len = 30
    
    src = torch.randint(4, config.vocab_size, (batch_size, src_len))
    trg = torch.randint(4, config.vocab_size, (batch_size, trg_len))
    src_lengths = torch.tensor([src_len] * batch_size)
    
    model.eval()
    with torch.no_grad():
        outputs = model(src, src_lengths, trg, teacher_forcing_ratio=0)
        print(f"Output shape: {outputs.shape}")  # [batch, trg_len, vocab_size]
        
        generated, _ = model.generate(src, src_lengths, max_length=50)
        print(f"Generated shape: {generated.shape}")
