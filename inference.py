"""
Inference script for the finetuned Qwen3 model.

Usage:
    python inference.py --model_path ./qwen3_finetuned/final
    python inference.py --model_path ./qwen3_finetuned/final --interactive
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_finetuned_model(
        args.model_path,
        args.base_model,
        args.device
    )
    
    if args.interactive:
        interactive_chat(model, tokenizer, device)
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
