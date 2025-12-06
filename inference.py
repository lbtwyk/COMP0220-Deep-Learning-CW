"""
Inference script for the finetuned Qwen3 model.

Usage:
    python inference.py --model_path ./qwen3_finetuned/final
    python inference.py --model_path ./qwen3_finetuned/final --interactive
"""

import argparse
from pathlib import Path
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel
from collections import Counter
import json
import random


def load_finetuned_model(
    model_path: str,
    base_model: str = None,
    device: str = "auto"
):
    """
    Load the finetuned model for inference.
    
    Args:
        model_path: Path to the finetuned model (LoRA adapter or full model)
        base_model: Base model name (required if model_path is LoRA adapter)
        device: Device to load model on ("auto", "cuda", "mps", "cpu")
    """
    model_path = Path(model_path)
    
    # Check if this is a LoRA adapter or full model
    adapter_config = model_path / "adapter_config.json"
    is_lora = adapter_config.exists()
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}")
    print(f"LoRA adapter: {is_lora}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_lora:
        # Load base model + LoRA adapter
        if base_model is None:
            # Try to infer base model from adapter config
            import json
            with open(adapter_config) as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path")
            if not base_model or (os.path.isabs(base_model) and not Path(base_model).exists()):
                base_model = "Qwen/Qwen3-4B-Instruct-2507"
        
        if base_model is None:
            raise ValueError("Base model not specified and could not be inferred")
        
        print(f"Loading base model: {base_model}")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "mps" else None,
            trust_remote_code=True,
        )
        
        if device == "mps":
            model = model.to("mps")
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
        
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "mps" else None,
            trust_remote_code=True,
        )
        
        if device == "mps":
            model = model.to("mps")
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    device: str,
    user_prompt: str,
    system_prompt: str = "You are a friendly tutor who explains sign languages and Deaf culture clearly.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a response from the model."""
    # If temperature <= 0, switch to greedy (no sampling) to avoid logits processor error
    if temperature is not None and temperature <= 0:
        do_sample = False
        temperature = None
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    
    # Move to device
    if device == "mps":
        inputs = inputs.to("mps")
    elif device == "cuda":
        inputs = inputs.to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response (only the generated part)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    return response.strip()


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    return " ".join(text.split())


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_text(prediction) == _normalize_text(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    gt_tokens = _normalize_text(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def load_embedding_model(model_name: str, device: str):
    """Load a lightweight embedding model for semantic similarity."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if device in {"cuda", "mps"}:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def embed_texts(texts, tokenizer, model, device: str):
    """Compute mean pooled embeddings."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    if device == "cuda":
        enc = {k: v.cuda() for k, v in enc.items()}
    elif device == "mps":
        enc = {k: v.to("mps") for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.cpu()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum().item())


def interactive_chat(model, tokenizer, device: str):
    """Run interactive chat session."""
    print("\n" + "=" * 60)
    print("Interactive Chat Mode")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'system: <prompt>' to change the system prompt")
    print("=" * 60 + "\n")
    
    system_prompt = "You are a friendly tutor who explains sign languages and Deaf culture clearly."
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            
            if user_input.lower().startswith("system:"):
                system_prompt = user_input[7:].strip()
                print(f"System prompt updated to: {system_prompt}")
                continue
            
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(
                model, tokenizer, device,
                user_prompt=user_input,
                system_prompt=system_prompt,
            )
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def load_qa_dataset(dataset_path: str):
    """Load a QA-style dataset from JSON (knowledge_dataset.json or train.json).

    Returns a list of dicts with keys: user_prompt, answer, system.
    """
    dataset_path = Path(dataset_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        if isinstance(item, dict) and "input" in item and "output" in item:
            # knowledge_dataset.json format
            system_prompt = "You are a Deaf culture specialist and ASL tutor. Answer clearly."
            examples.append(
                {
                    "user_prompt": item["input"],
                    "answer": item["output"],
                    "system": system_prompt,
                }
            )
        elif isinstance(item, dict) and "question" in item and "answer" in item:
            # train.json format
            question = item["question"]
            # Ignore context to match training format
            user_prompt = question
            examples.append(
                {
                    "user_prompt": user_prompt,
                    "answer": item["answer"],
                    "system": "You are a Deaf culture specialist and ASL tutor. Answer clearly.",
                }
            )
    
    return examples


def evaluate_model_on_dataset(
    model,
    tokenizer,
    device: str,
    dataset_path: str,
    num_samples: int = 20,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    embedding_model_name: str = "intfloat/e5-small-v2",
):
    """Run evaluation on a JSON dataset and report simple metrics."""
    examples = load_qa_dataset(dataset_path)
    if not examples:
        print(f"No usable examples found in {dataset_path}")
        return
    
    # Load embedding model once
    try:
        emb_model, emb_tokenizer = load_embedding_model(embedding_model_name, device)
        use_semantic = True
        print(f"Loaded embedding model for semantic similarity: {embedding_model_name}")
    except Exception as e:
        print(f"Warning: semantic embedding model load failed ({e}); skipping semantic similarity.")
        emb_model, emb_tokenizer, use_semantic = None, None, False
    
    if num_samples is not None and num_samples > 0 and len(examples) > num_samples:
        examples = random.sample(examples, num_samples)
    
    total_em = 0.0
    total_f1 = 0.0
    total_sem = 0.0
    
    for idx, ex in enumerate(examples, start=1):
        user_prompt = ex["user_prompt"]
        gold = ex["answer"]
        system_prompt = ex["system"]
        
        prediction = generate_response(
            model,
            tokenizer,
            device,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        em = compute_exact_match(prediction, gold)
        f1 = compute_f1(prediction, gold)
        sem = None
        if use_semantic:
            embs = embed_texts([prediction, gold], emb_tokenizer, emb_model, device)
            sem = cosine_similarity(embs[0], embs[1])
            total_sem += sem
        
        total_em += em
        total_f1 += f1
        
        print("\n" + "=" * 80)
        print(f"Example {idx}")
        print("-" * 80)
        print(f"User prompt: {user_prompt}")
        print(f"Ground truth: {gold}")
        print(f"Model output: {prediction}")
        if sem is not None:
            print(f"Exact match: {em:.3f} | F1: {f1:.3f} | Semantic cosine: {sem:.3f}")
        else:
            print(f"Exact match: {em:.3f} | F1: {f1:.3f}")
    
    n = len(examples)
    print("\n" + "=" * 80)
    print("Evaluation summary")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Examples evaluated: {n}")
    print(f"Average exact match: {total_em / n:.3f}")
    print(f"Average F1: {total_f1 / n:.3f}")
    if use_semantic:
        print(f"Average semantic cosine: {total_sem / n:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Inference with finetuned Qwen3 model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the finetuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (required if model_path is LoRA adapter)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive chat mode"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate response for"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to a JSON dataset file (e.g. knowledge_dataset.json or train.json) for evaluation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of random examples to evaluate from the dataset (0 = all)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="intfloat/e5-small-v2",
        help="Embedding model for semantic similarity scoring",
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_finetuned_model(
        args.model_path,
        args.base_model,
        args.device
    )
    
    if args.interactive:
        interactive_chat(model, tokenizer, device)
    elif args.dataset_path:
        evaluate_model_on_dataset(
            model,
            tokenizer,
            device,
            dataset_path=args.dataset_path,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            embedding_model_name=args.embedding_model,
        )
    elif args.prompt:
        response = generate_response(
            model, tokenizer, device,
            user_prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"\nResponse: {response}")
    else:
        # Demo with sample questions
        sample_questions = [
            "What is ASL?",
            "Why do some Deaf people prefer the term 'Deaf' with a capital D?",
            "What are the 5 parameters of an ASL sign?",
        ]
        
        print("\n" + "=" * 60)
        print("Demo: Sample Questions")
        print("=" * 60)
        
        for question in sample_questions:
            print(f"\n{'='*40}")
            print(f"Q: {question}")
            print(f"{'='*40}")
            response = generate_response(
                model, tokenizer, device,
                user_prompt=question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print(f"A: {response}")


if __name__ == "__main__":
    main()
