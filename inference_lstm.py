"""
LSTM Inference Script.

Load trained LSTM model and generate responses for evaluation.
Supports both interactive mode and batch evaluation.

Usage:
    python inference_lstm.py --checkpoint ./lstm_baseline/checkpoint_best.pt
    python inference_lstm.py --checkpoint ./lstm_baseline/checkpoint_best.pt --eval
    python inference_lstm.py --checkpoint ./lstm_baseline/checkpoint_best.pt --interactive
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import time

import torch
from tqdm import tqdm

from lstm_model import LSTMConfig, Seq2SeqLSTM, create_model
from lstm_data import Vocabulary, load_raw_data


def load_model(
    checkpoint_path: Path,
    vocab_path: Optional[Path] = None,
    device: str = "cpu"
) -> tuple:
    """
    Load trained LSTM model and vocabulary.
    
    Args:
        checkpoint_path: Path to model checkpoint
        vocab_path: Path to vocabulary file (defaults to same dir as checkpoint)
        device: Device to load model on
    
    Returns:
        model, vocab, config
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    config = LSTMConfig()
    for key, value in checkpoint["config"].items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.device = device
    
    # Create and load model
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load vocabulary
    if vocab_path is None:
        vocab_path = checkpoint_path.parent / "vocab.json"
    vocab = Vocabulary.load(vocab_path)
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    return model, vocab, config


def generate_response(
    model: Seq2SeqLSTM,
    vocab: Vocabulary,
    question: str,
    config: LSTMConfig,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9
) -> str:
    """
    Generate response for a single question.
    
    Args:
        model: Trained LSTM model
        vocab: Vocabulary
        question: Input question
        config: Model config
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling threshold
    
    Returns:
        Generated response string
    """
    model.eval()
    
    # Encode question
    src_ids = vocab.encode(question)
    src_ids = src_ids[:config.max_input_length]
    src = torch.tensor([src_ids], dtype=torch.long, device=config.device)
    src_len = torch.tensor([len(src_ids)])
    
    # Generate
    with torch.no_grad():
        generated, _ = model.generate(
            src, src_len,
            max_length=config.max_output_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode
    response = vocab.decode(generated[0].tolist())
    return response


def batch_generate(
    model: Seq2SeqLSTM,
    vocab: Vocabulary,
    questions: List[str],
    config: LSTMConfig,
    batch_size: int = 16,
    temperature: float = 0.7
) -> List[str]:
    """Generate responses for multiple questions."""
    model.eval()
    responses = []
    
    for i in tqdm(range(0, len(questions), batch_size), desc="Generating"):
        batch_questions = questions[i:i + batch_size]
        
        # Encode batch
        batch_src = []
        batch_lens = []
        max_len = 0
        
        for q in batch_questions:
            ids = vocab.encode(q)[:config.max_input_length]
            batch_src.append(ids)
            batch_lens.append(len(ids))
            max_len = max(max_len, len(ids))
        
        # Pad
        padded = []
        for ids in batch_src:
            padded.append(ids + [vocab.word2idx[vocab.PAD_TOKEN]] * (max_len - len(ids)))
        
        src = torch.tensor(padded, dtype=torch.long, device=config.device)
        src_len = torch.tensor(batch_lens)
        
        # Generate
        with torch.no_grad():
            generated, _ = model.generate(
                src, src_len,
                max_length=config.max_output_length,
                temperature=temperature
            )
        
        # Decode
        for gen in generated:
            response = vocab.decode(gen.tolist())
            responses.append(response)
    
    return responses


def evaluate_model(
    model: Seq2SeqLSTM,
    vocab: Vocabulary,
    config: LSTMConfig,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Evaluate model on test data and compute metrics.
    
    Returns evaluation results including sample predictions.
    """
    base_dir = Path(__file__).parent
    
    # Load test data (use knowledge dataset)
    questions, references = load_raw_data(base_dir, use_knowledge=True, use_education=False)
    
    # Use a subset for evaluation
    eval_size = min(100, len(questions))
    questions = questions[:eval_size]
    references = references[:eval_size]
    
    print(f"\nEvaluating on {eval_size} samples...")
    
    # Generate predictions
    start_time = time.time()
    predictions = batch_generate(model, vocab, questions, config, temperature=0.7)
    gen_time = time.time() - start_time
    
    # Compute simple metrics
    exact_match = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    
    # Token overlap (simple F1-like metric)
    def token_overlap(pred: str, ref: str) -> float:
        pred_tokens = set(vocab.tokenize(pred))
        ref_tokens = set(vocab.tokenize(ref))
        if not pred_tokens or not ref_tokens:
            return 0.0
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens) if pred_tokens else 0
        recall = overlap / len(ref_tokens) if ref_tokens else 0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    f1_scores = [token_overlap(p, r) for p, r in zip(predictions, references)]
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    # Compute average response length
    avg_pred_len = sum(len(p.split()) for p in predictions) / len(predictions)
    avg_ref_len = sum(len(r.split()) for r in references) / len(references)
    
    results = {
        "num_samples": eval_size,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / eval_size,
        "avg_token_f1": avg_f1,
        "avg_pred_length": avg_pred_len,
        "avg_ref_length": avg_ref_len,
        "generation_time_sec": gen_time,
        "samples_per_sec": eval_size / gen_time,
        "samples": []
    }
    
    # Add sample predictions
    for i in range(min(10, eval_size)):
        results["samples"].append({
            "question": questions[i],
            "reference": references[i],
            "prediction": predictions[i],
            "token_f1": f1_scores[i]
        })
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Samples evaluated: {eval_size}")
    print(f"Exact match rate: {results['exact_match_rate']:.2%}")
    print(f"Average token F1: {results['avg_token_f1']:.4f}")
    print(f"Avg prediction length: {results['avg_pred_length']:.1f} words")
    print(f"Avg reference length: {results['avg_ref_length']:.1f} words")
    print(f"Generation speed: {results['samples_per_sec']:.2f} samples/sec")
    
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*60}")
    for s in results["samples"][:5]:
        print(f"\nQ: {s['question'][:100]}...")
        print(f"R: {s['reference'][:100]}...")
        print(f"P: {s['prediction'][:100]}...")
        print(f"F1: {s['token_f1']:.3f}")
    
    # Save results
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
    
    return results


def interactive_mode(
    model: Seq2SeqLSTM,
    vocab: Vocabulary,
    config: LSTMConfig
):
    """Interactive chat mode."""
    print(f"\n{'='*60}")
    print("LSTM INTERACTIVE MODE")
    print("Type 'quit' or 'exit' to stop")
    print(f"{'='*60}\n")
    
    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not question:
                continue
            
            start = time.time()
            response = generate_response(model, vocab, question, config)
            elapsed = time.time() - start
            
            print(f"LSTM: {response}")
            print(f"  (generated in {elapsed:.2f}s)\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="LSTM Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, default=None,
                        help="Path to vocabulary file (default: same dir as checkpoint)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "mps", "cpu"],
                        help="Device to use")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation on test set")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive chat mode")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for evaluation results")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--question", type=str, default=None,
                        help="Single question to answer")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    vocab_path = Path(args.vocab) if args.vocab else None
    
    model, vocab, config = load_model(checkpoint_path, vocab_path, device)
    
    # Run requested mode
    if args.question:
        response = generate_response(model, vocab, args.question, config, temperature=args.temperature)
        print(f"\nQuestion: {args.question}")
        print(f"Response: {response}")
    
    elif args.eval:
        output_path = Path(args.output) if args.output else checkpoint_path.parent / "eval_results.json"
        evaluate_model(model, vocab, config, output_path)
    
    elif args.interactive:
        interactive_mode(model, vocab, config)
    
    else:
        # Default: show a few sample generations
        print("\nGenerating sample responses...")
        sample_questions = [
            "What does CODA stand for?",
            "Why is eye contact important in ASL?",
            "What is Deaf culture?",
        ]
        
        for q in sample_questions:
            response = generate_response(model, vocab, q, config)
            print(f"\nQ: {q}")
            print(f"A: {response}")


if __name__ == "__main__":
    main()
