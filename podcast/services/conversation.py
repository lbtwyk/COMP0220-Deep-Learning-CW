"""
Conversation Memory

Tracks the conversation history for the podcast session.
Provides context for agent responses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    agent: str              # "rick", "morty", "summer", or "user"
    content: str            # What was said
    timestamp: datetime = field(default_factory=datetime.now)
    audio_url: Optional[str] = None  # URL to audio if generated
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent": self.agent,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "audio_url": self.audio_url,
            "metadata": self.metadata,
        }
    
    def to_message(self) -> Dict[str, str]:
        """Convert to LLM message format."""
        role = "user" if self.agent == "user" else "assistant"
        # Don't add agent tags - the LLM context will handle multi-agent conversations
        # The agent name is already stored in self.agent
        content = self.content
        return {"role": role, "content": content}


class ConversationMemory:
    """
    Manages conversation history for a podcast session.
    
    Features:
    - Stores all turns with metadata
    - Provides context window for LLM
    - Supports topic-based segmentation
    - Can export transcript
    """
    
    def __init__(self, max_context_turns: int = 20):
        """
        Initialize conversation memory.
        
        Args:
            max_context_turns: Maximum turns to include in LLM context
        """
        self.max_context_turns = max_context_turns
        self.turns: List[ConversationTurn] = []
        self.topics: List[Dict[str, Any]] = []  # Track topic changes
        self.session_start: datetime = datetime.now()
    
    def add_turn(
        self,
        agent: str,
        content: str,
        audio_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a turn to the conversation."""
        turn = ConversationTurn(
            agent=agent,
            content=content,
            audio_url=audio_url,
            metadata=metadata or {},
        )
        self.turns.append(turn)
    
    def add_topic(self, topic: str):
        """Mark a new topic in the conversation."""
        self.topics.append({
            "topic": topic,
            "started_at": datetime.now().isoformat(),
            "turn_index": len(self.turns),
        })
    
    def get_context(self, num_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation context for LLM.
        
        Args:
            num_turns: Number of recent turns to include (None = use max)
            
        Returns:
            List of messages in LLM format
        """
        n = num_turns or self.max_context_turns
        recent_turns = self.turns[-n:]
        return [turn.to_message() for turn in recent_turns]
    
    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get the N most recent turns."""
        return self.turns[-n:]
    
    def get_last_speaker(self) -> Optional[str]:
        """Get the agent who spoke last."""
        if self.turns:
            return self.turns[-1].agent
        return None
    
    def get_turn_count(self) -> int:
        """Get total number of turns."""
        return len(self.turns)
    
    def get_topic_turn_count(self) -> int:
        """Get turns since the last topic change."""
        if not self.topics:
            return len(self.turns)
        
        last_topic_index = self.topics[-1]["turn_index"]
        return len(self.turns) - last_topic_index
    
    def clear(self):
        """Clear all conversation history."""
        self.turns = []
        self.topics = []
        self.session_start = datetime.now()
    
    def to_transcript(self) -> str:
        """Export conversation as a readable transcript."""
        lines = []
        lines.append(f"# SignTutor Podcast Transcript")
        lines.append(f"Session started: {self.session_start.isoformat()}")
        lines.append("")
        
        current_topic = None
        topic_idx = 0
        
        for i, turn in enumerate(self.turns):
            # Check if we hit a new topic
            while topic_idx < len(self.topics) and self.topics[topic_idx]["turn_index"] == i:
                current_topic = self.topics[topic_idx]["topic"]
                lines.append(f"\n## Topic: {current_topic}\n")
                topic_idx += 1
            
            # Format the turn
            emoji = {"rick": "🥒", "morty": "😰", "summer": "🎯", "user": "👤"}.get(turn.agent, "")
            name = turn.agent.upper()
            timestamp = turn.timestamp.strftime("%H:%M:%S")
            lines.append(f"**{emoji} {name}** ({timestamp}): {turn.content}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "topics": self.topics,
            "session_start": self.session_start.isoformat(),
            "turn_count": len(self.turns),
        }
    
    def export_json(self) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=2)

