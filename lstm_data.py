"""
Data preprocessing for LSTM model.

Builds vocabulary from scratch and creates PyTorch datasets.
Uses the same underlying data as Qwen3 but with character/word-level tokenization.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from lstm_model import LSTMConfig


@dataclass
class Vocabulary:
    """Simple word-level vocabulary."""
    word2idx: Dict[str, int]
    idx2word: Dict[int, str]
    word_freq: Counter
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    @classmethod
    def build(cls, texts: List[str], min_freq: int = 2, max_vocab_size: int = 10000) -> "Vocabulary":
        """Build vocabulary from list of texts."""
        word_freq = Counter()
        
        for text in texts:
            tokens = cls.tokenize(text)
            word_freq.update(tokens)
        
        # Start with special tokens
        word2idx = {
            cls.PAD_TOKEN: 0,
            cls.SOS_TOKEN: 1,
            cls.EOS_TOKEN: 2,
            cls.UNK_TOKEN: 3,
        }
        
        # Add frequent words
        for word, freq in word_freq.most_common(max_vocab_size - 4):
            if freq >= min_freq:
                word2idx[word] = len(word2idx)
        
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        return cls(word2idx=word2idx, idx2word=idx2word, word_freq=word_freq)
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple word tokenization with basic preprocessing."""
        # Lowercase and basic cleaning
        text = text.lower().strip()
        # Split on whitespace and punctuation, keeping punctuation as tokens
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token ids."""
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]
        
        if add_special_tokens:
            ids = [self.word2idx[self.SOS_TOKEN]] + ids + [self.word2idx[self.EOS_TOKEN]]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token ids back to text."""
        special_ids = {0, 1, 2} if skip_special_tokens else set()
        tokens = [self.idx2word.get(i, self.UNK_TOKEN) for i in ids if i not in special_ids]
        
        # Simple detokenization
        text = ""
        for token in tokens:
            if token in ".,!?;:":
                text += token
            elif text and not text.endswith(" "):
                text += " " + token
            else:
                text += token
        
        return text.strip()
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def save(self, path: Path):
        """Save vocabulary to file."""
        data = {
            "word2idx": self.word2idx,
            "word_freq": dict(self.word_freq),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        word2idx = data["word2idx"]
        idx2word = {int(idx): word for word, idx in word2idx.items()}
        word_freq = Counter(data.get("word_freq", {}))
        
        return cls(word2idx=word2idx, idx2word=idx2word, word_freq=word_freq)


class QADataset(Dataset):
    """PyTorch dataset for question-answer pairs."""
    
    def __init__(
        self,
        questions: List[str],
        answers: List[str],
        vocab: Vocabulary,
        max_input_length: int = 128,
        max_output_length: int = 256
    ):
        self.questions = questions
        self.answers = answers
        self.vocab = vocab
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # Encode
        src_ids = self.vocab.encode(question)
        trg_ids = self.vocab.encode(answer)
        
        # Truncate if needed
        src_ids = src_ids[:self.max_input_length]
        trg_ids = trg_ids[:self.max_output_length]
        
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "trg": torch.tensor(trg_ids, dtype=torch.long),
            "src_len": len(src_ids),
            "trg_len": len(trg_ids),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader with padding."""
    src_list = [item["src"] for item in batch]
    trg_list = [item["trg"] for item in batch]
    src_lens = torch.tensor([item["src_len"] for item in batch])
    trg_lens = torch.tensor([item["trg_len"] for item in batch])
    
    # Pad sequences
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=0)
    trg_padded = pad_sequence(trg_list, batch_first=True, padding_value=0)
    
    return {
        "src": src_padded,
        "trg": trg_padded,
        "src_len": src_lens,
        "trg_len": trg_lens,
    }


def load_raw_data(base_dir: Path, use_knowledge: bool = True, use_education: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load raw question-answer pairs from datasets.
    
    For LSTM baseline, we use a subset of data:
    - Knowledge dataset: ~100 samples (good for from-scratch training)
    - Education dialogue: Optional, adds more data but may be noisy
    
    Returns:
        questions: list of input questions
        answers: list of target answers
    """
    questions = []
    answers = []
    
    if use_knowledge:
        knowledge_dir = base_dir / "knowledge_dataset"
        if knowledge_dir.exists():
            for json_file in sorted(knowledge_dir.glob("*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    for item in data:
                        # Handle knowledge format (input/output)
                        if "input" in item and "output" in item:
                            q = item["input"].strip()
                            a = item["output"].strip()
                            if q and a:
                                questions.append(q)
                                answers.append(a)
                        # Handle QA format (question/answer)
                        elif "question" in item and "answer" in item:
                            q = item["question"].strip()
                            a = item["answer"].strip()
                            if q and a:
                                questions.append(q)
                                answers.append(a)
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}")
    
    if use_education:
        education_dir = base_dir / "Education-Dialogue-Dataset-main"
        if education_dir.exists():
            # Load only first train file to limit data size
            for json_file in sorted(education_dir.glob("conversations_train*.json"))[:1]:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    for item in data:
                        conv = item.get("conversation", [])
                        # Extract student-teacher pairs
                        for i in range(len(conv) - 1):
                            if conv[i].get("role", "").lower() == "student":
                                if conv[i + 1].get("role", "").lower() == "teacher":
                                    q = conv[i].get("text", "").strip()
                                    a = conv[i + 1].get("text", "").strip()
                                    if q and a and len(q) > 5 and len(a) > 5:
                                        questions.append(q)
                                        answers.append(a)
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}")
    
    return questions, answers


def prepare_data(
    base_dir: Path,
    config: LSTMConfig,
    use_knowledge: bool = True,
    use_education: bool = False,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """
    Prepare data loaders for training.
    
    Args:
        base_dir: Base directory containing datasets
        config: LSTM configuration
        use_knowledge: Include knowledge dataset
        use_education: Include education dialogue dataset
        val_split: Validation split ratio
        seed: Random seed
    
    Returns:
        train_loader, val_loader, vocabulary
    """
    print("Loading raw data...")
    questions, answers = load_raw_data(base_dir, use_knowledge, use_education)
    print(f"Loaded {len(questions)} QA pairs")
    
    # Build vocabulary from all texts
    print("Building vocabulary...")
    all_texts = questions + answers
    vocab = Vocabulary.build(
        all_texts,
        min_freq=config.min_word_freq,
        max_vocab_size=config.vocab_size
    )
    print(f"Vocabulary size: {len(vocab)}")
    
    # Split data
    torch.manual_seed(seed)
    indices = torch.randperm(len(questions)).tolist()
    val_size = int(len(questions) * val_split)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_questions = [questions[i] for i in train_indices]
    train_answers = [answers[i] for i in train_indices]
    val_questions = [questions[i] for i in val_indices]
    val_answers = [answers[i] for i in val_indices]
    
    print(f"Train size: {len(train_questions)}, Val size: {len(val_questions)}")
    
    # Create datasets
    train_dataset = QADataset(
        train_questions, train_answers, vocab,
        max_input_length=config.max_input_length,
        max_output_length=config.max_output_length
    )
    val_dataset = QADataset(
        val_questions, val_answers, vocab,
        max_input_length=config.max_input_length,
        max_output_length=config.max_output_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if config.device != "cpu" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if config.device != "cpu" else False
    )
    
    return train_loader, val_loader, vocab


if __name__ == "__main__":
    # Test data loading
    base_dir = Path(__file__).parent
    config = LSTMConfig()
    
    train_loader, val_loader, vocab = prepare_data(
        base_dir, config,
        use_knowledge=True,
        use_education=False
    )
    
    print(f"\nVocabulary examples:")
    for i in range(10):
        print(f"  {i}: {vocab.idx2word[i]}")
    
    print(f"\nSample batch:")
    batch = next(iter(train_loader))
    print(f"  src shape: {batch['src'].shape}")
    print(f"  trg shape: {batch['trg'].shape}")
    
    # Decode a sample
    src_text = vocab.decode(batch['src'][0].tolist())
    trg_text = vocab.decode(batch['trg'][0].tolist())
    print(f"\n  Question: {src_text}")
    print(f"  Answer: {trg_text}")
