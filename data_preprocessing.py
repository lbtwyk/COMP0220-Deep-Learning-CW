"""
Data preprocessing module for Qwen3 finetuning.
Unifies three dataset formats into a single conversational format.

Datasets:
1. knowledge_dataset.json - system/input/output format (Deaf culture & ASL)
2. train.json - context/question/answer format (Sign language QA)
3. Education-Dialogue-Dataset - multi-turn teacher/student conversations
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset, concatenate_datasets


SYSTEM_PROMPT = "You are a Deaf culture specialist and ASL tutor. Answer clearly."


def load_knowledge_dataset(path: Path) -> List[Dict[str, Any]]:
    """
    Load knowledge-style data with format:
    {"system": "...", "input": "...", "output": "..."} or {"input": "...", "output": "..."}
    
    Convert to conversational format:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    conversations = []
    for item in data:
        messages = []
        
        # Use unified system prompt for all datasets
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
        # Add user input
        if item.get("input"):
            messages.append({"role": "user", "content": item["input"]})
        
        # Add assistant output
        if item.get("output"):
            messages.append({"role": "assistant", "content": item["output"]})
        
        if len(messages) >= 2:  # At least user + assistant
            conversations.append({"messages": messages})
    
    return conversations


def load_knowledge_directory(
    dir_path: Path,
    include_knowledge: bool,
    include_qa: bool
) -> List[Dict[str, Any]]:
    """
    Load every JSON file under knowledge_dataset/ and convert by format:
    - knowledge format: {"input", "output"} (optional "system")
    - QA format: {"question", "answer"} (optional "context")
    
    Flags control which formats are included so we don't mix in unwanted data.
    """
    conversations: List[Dict[str, Any]] = []
    
    if not dir_path.exists():
        return conversations
    
    print(f"Scanning knowledge datasets in {dir_path}...")
    for file_path in sorted(dir_path.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Skipping {file_path.name}: failed to load ({e})")
            continue
        
        if not isinstance(data, list) or not data:
            print(f"  Skipping {file_path.name}: not a non-empty list")
            continue
        
        first_item = data[0]
        
        if include_knowledge and "input" in first_item and "output" in first_item:
            file_convs = load_knowledge_dataset(file_path)
            print(f"  Loaded {len(file_convs)} knowledge samples from {file_path.name}")
            conversations.extend(file_convs)
        elif include_qa and "question" in first_item and "answer" in first_item:
            file_convs = load_qa_dataset(file_path)
            print(f"  Loaded {len(file_convs)} QA samples from {file_path.name}")
            conversations.extend(file_convs)
        else:
            print(f"  Skipping {file_path.name}: format not included by flags")
    
    return conversations


def load_qa_dataset(path: Path) -> List[Dict[str, Any]]:
    """
    Load train.json with format:
    {"context": "...", "question": "...", "answer": "..."}
    
    Convert to conversational format with context in system prompt.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    conversations = []
    for item in data:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        
        # Skip malformed entries
        if not question or not answer:
            continue
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Do not embed context in the user turn; keep prompts consistent across datasets
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
        
        conversations.append({"messages": messages})
    
    return conversations


def load_education_dialogue_dataset(
    data_dir: Path,
    split: str = "train",
    max_files: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load Education Dialogue Dataset with format:
    {
        "background_info": {"topic": "...", ...},
        "conversation": [{"role": "Teacher/Student", "text": "..."}]
    }
    
    Convert to conversational format, mapping Teacher -> assistant, Student -> user.
    """
    if split == "train":
        pattern = "conversations_train*.json"
    else:
        pattern = "conversations_eval.json"
    
    files = sorted(data_dir.glob(pattern))
    if max_files:
        files = files[:max_files]
    
    conversations = []
    
    for file_path in files:
        print(f"Loading {file_path.name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data:
            background = item.get("background_info", {})
            topic = background.get("topic", "")
            conv = item.get("conversation", [])
            
            if not conv:
                continue
            
            # Create system message with topic context
            system_content = (
                f"You are a helpful teacher explaining the topic: {topic}. "
                "Adapt your teaching style to the student's needs and preferences."
            )
            
            messages = [{"role": "system", "content": system_content}]
            
            for turn in conv:
                role = turn.get("role", "").strip()
                text = turn.get("text", "").strip()
                
                if not text:
                    continue
                
                # Map Teacher -> assistant, Student -> user
                if role.lower() == "teacher":
                    messages.append({"role": "assistant", "content": text})
                elif role.lower() == "student":
                    messages.append({"role": "user", "content": text})
            
            # Ensure conversation has proper structure (at least one user and one assistant)
            has_user = any(m["role"] == "user" for m in messages)
            has_assistant = any(m["role"] == "assistant" for m in messages)
            
            if has_user and has_assistant:
                conversations.append({"messages": messages})
    
    return conversations


def create_unified_dataset(
    base_dir: Path,
    include_knowledge: bool = True,
    include_qa: bool = True,
    include_education: bool = True,
    education_max_files: Optional[int] = None,
    education_split: str = "train"
) -> Dataset:
    """
    Create a unified HuggingFace Dataset from all data sources.
    
    Args:
        base_dir: Base directory containing all datasets
        include_knowledge: Include knowledge_dataset.json
        include_qa: Include train.json
        include_education: Include Education Dialogue Dataset
        education_max_files: Limit number of education dialogue files (for testing)
        education_split: "train" or "eval" for education dialogue
    
    Returns:
        HuggingFace Dataset with unified conversational format
    """
    all_conversations = []
    
    # Unified loading from knowledge_dataset directory (supports knowledge + QA formats)
    if include_knowledge or include_qa:
        knowledge_dir = base_dir / "knowledge_dataset"
        if knowledge_dir.exists():
            knowledge_convs = load_knowledge_directory(
                knowledge_dir,
                include_knowledge=include_knowledge,
                include_qa=include_qa,
            )
            print(f"Loaded {len(knowledge_convs)} conversations from knowledge_dataset/")
            all_conversations.extend(knowledge_convs)
        else:
            # Fallback to legacy single-file paths at root
            if include_knowledge:
                knowledge_path = base_dir / "knowledge_dataset.json"
                if knowledge_path.exists():
                    print(f"Loading knowledge dataset from {knowledge_path}...")
                    knowledge_data = load_knowledge_dataset(knowledge_path)
                    print(f"  Loaded {len(knowledge_data)} conversations")
                    all_conversations.extend(knowledge_data)
            if include_qa:
                qa_path = base_dir / "train.json"
                if qa_path.exists():
                    print(f"Loading QA dataset from {qa_path}...")
                    qa_data = load_qa_dataset(qa_path)
                    print(f"  Loaded {len(qa_data)} conversations")
                    all_conversations.extend(qa_data)
    
    if include_education:
        education_dir = base_dir / "Education-Dialogue-Dataset-main"
        if education_dir.exists():
            print(f"Loading education dialogue dataset from {education_dir}...")
            education_data = load_education_dialogue_dataset(
                education_dir,
                split=education_split,
                max_files=education_max_files
            )
            print(f"  Loaded {len(education_data)} conversations")
            all_conversations.extend(education_data)
    
    print(f"\nTotal conversations: {len(all_conversations)}")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(all_conversations)
    
    return dataset


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.1,
    seed: int = 42
) -> Dict[str, Dataset]:
    """
    Split dataset into train and validation sets.
    """
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return {
        "train": split["train"],
        "validation": split["test"]
    }


def preview_dataset(dataset: Dataset, num_samples: int = 3):
    """
    Preview a few samples from the dataset.
    """
    print("\n" + "=" * 60)
    print("Dataset Preview")
    print("=" * 60)
    
    for i, sample in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
        print(f"\n--- Sample {i + 1} ---")
        messages = sample["messages"]
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            print(f"[{role}]: {content}")
        print()


if __name__ == "__main__":
    # Test the data preprocessing
    base_dir = Path(__file__).parent
    
    # Create unified dataset (limit education files for quick testing)
    dataset = create_unified_dataset(
        base_dir,
        include_knowledge=True,
        include_qa=True,
        include_education=True,
        education_max_files=1  # Use only 1 file for testing
    )
    
    # Preview samples
    preview_dataset(dataset, num_samples=3)
    
    # Split into train/val
    splits = split_dataset(dataset)
    print(f"\nTrain size: {len(splits['train'])}")
    print(f"Validation size: {len(splits['validation'])}")
