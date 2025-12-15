"""
Inference script for the trained Dummy Transformer LM
"""

import torch
import json
import argparse

from model import DummyTransformerLM
from train import SimpleTokenizer


def load_model(checkpoint_path: str, tokenizer_path: str, device: str = 'cuda'):
    """Load trained model and tokenizer"""
    
    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = DummyTransformerLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=0.0  # No dropout during inference
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, 
                   max_new_tokens: int = 200,
                   temperature: float = 0.8,
                   device: str = 'cuda'):
    """Generate answer for a given question"""
    
    # Format input: <bos> question <sep>
    bos = [tokenizer.special_tokens['<bos>']]
    sep = [tokenizer.special_tokens['<sep>']]
    q_tokens = tokenizer.encode(question)
    
    input_ids = torch.tensor([bos + q_tokens + sep], dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50
        )
    
    # Decode (skip input tokens)
    generated = output_ids[0, len(input_ids[0]):].tolist()
    
    # Stop at <eos> if found
    eos_id = tokenizer.special_tokens['<eos>']
    if eos_id in generated:
        generated = generated[:generated.index(eos_id)]
    
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description='Generate with Dummy LM')
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str,
                       default='./checkpoints/tokenizer.json',
                       help='Path to tokenizer')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    print(f"Loaded model with {model.count_parameters():,} parameters")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Dummy Transformer LM - Interactive Mode")
    print("Type a question about ASL/Deaf culture, or 'quit' to exit")
    print("=" * 60 + "\n")
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        answer = generate_answer(
            model, tokenizer, question,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device
        )
        
        print(f"\nAnswer: {answer}\n")


if __name__ == '__main__':
    main()
