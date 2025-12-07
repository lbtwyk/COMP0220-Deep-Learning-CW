import json
import random
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Swapping to Qwen3 4B Instruct (2507 release) per latest requirement
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "mps" else torch.float32,
        device_map={"": device},
    )
    return tokenizer, model, device

def chat_once(tokenizer, model, device, user_prompt: str):
    system_prompt = "You are a friendly tutor who explains sign languages and Deaf culture clearly."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 使用官方推荐的 chat template（Qwen3 model card 中提供）
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def pick_random_question(dataset_path: Path) -> str:
    with dataset_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    candidates = [item.get("question", "").strip() for item in records if item.get("question")]
    if not candidates:
        raise ValueError(f"No question entries found in {dataset_path}")

    return random.choice(candidates)

if __name__ == "__main__":
    tokenizer, model, device = load_model()
    print("Using device:", device)

    dataset_path = Path(__file__).with_name("train.json")
    user_prompt = pick_random_question(dataset_path)
    print(f"Randomly selected question: {user_prompt}")
    answer = chat_once(tokenizer, model, device, user_prompt)
    print("\n=== Model output ===\n")
    print(answer)
