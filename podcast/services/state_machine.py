"""
Podcast State Machine

Manages the podcast conversation flow and state transitions.

States:
- IDLE: Waiting for user to start
- WELCOME: Initial greeting, asking for topic
- TOPIC_INPUT: Waiting for user to provide topic
- DISCUSSING: Active discussion between agents
- MORTY_TURN: Morty (host) is speaking
- RICK_TURN: Rick (expert) is speaking
- USER_INTERRUPT: Handling user interruption
- NEW_TOPIC: Asking if user wants a new topic
- WRAP_UP: Ending the podcast
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
from datetime import datetime
import asyncio


class PodcastState(Enum):
    """Possible states of the podcast."""
    IDLE = auto()           # Not started
    WELCOME = auto()        # Greeting user
    TOPIC_INPUT = auto()    # Waiting for topic
    DISCUSSING = auto()     # General discussion state
    MORTY_TURN = auto()     # Morty speaking
    RICK_TURN = auto()      # Rick speaking  
    SUMMER_TURN = auto()    # Summer (coordinator) speaking
    USER_INTERRUPT = auto() # Handling user interrupt
    NEW_TOPIC = auto()      # Asking for new topic
    WRAP_UP = auto()        # Ending podcast
    PAUSED = auto()         # Temporarily paused


@dataclass
class PodcastContext:
    """Context data for the podcast session."""
    current_topic: str = ""                    # Current discussion topic
    topics_discussed: List[str] = field(default_factory=list)  # History of topics
    turn_count: int = 0                        # Number of turns in current topic
    max_turns_per_topic: int = 10              # Max turns before suggesting new topic
    last_speaker: str = ""                     # Who spoke last ("rick", "morty", "summer")
    pending_user_input: Optional[str] = None   # User input waiting to be processed
    pending_sign_input: Optional[str] = None   # Detected sign waiting to be processed
    started_at: Optional[datetime] = None      # When podcast started
    subtopics: List[str] = field(default_factory=list)  # Subtopics to cover


class PodcastStateMachine:
    """
    Manages podcast state and transitions.
    
    The state machine controls the flow of conversation:
    1. User joins → WELCOME
    2. Ask for topic → TOPIC_INPUT
    3. Topic received → DISCUSSING (alternating MORTY_TURN ↔ RICK_TURN)
    4. User interrupts → USER_INTERRUPT → back to DISCUSSING
    5. Topic exhausted → NEW_TOPIC
    6. User ends → WRAP_UP → IDLE
    """
    
    def __init__(self):
        self.state = PodcastState.IDLE
        self.context = PodcastContext()
        self._state_handlers: Dict[PodcastState, Callable] = {}
        self._transition_callbacks: List[Callable] = []
    
    def reset(self):
        """Reset the state machine to initial state."""
        self.state = PodcastState.IDLE
        self.context = PodcastContext()
    
    def start(self) -> PodcastState:
        """Start the podcast session."""
        self.state = PodcastState.WELCOME
        self.context.started_at = datetime.now()
        return self.state
    
    def set_topic(self, topic: str):
        """Set the current discussion topic."""
        if self.context.current_topic:
            self.context.topics_discussed.append(self.context.current_topic)
        self.context.current_topic = topic
        self.context.turn_count = 0
        self.context.subtopics = []
        self.state = PodcastState.DISCUSSING
    
    def next_turn(self) -> PodcastState:
        """
        Advance to the next speaker's turn.
        Alternates between Morty and Rick, with occasional Summer interjections.
        """
        self.context.turn_count += 1
        
        # Check if we should suggest a new topic
        if self.context.turn_count >= self.context.max_turns_per_topic:
            self.state = PodcastState.NEW_TOPIC
            return self.state
        
        # Alternate between speakers
        if self.context.last_speaker == "rick":
            self.state = PodcastState.MORTY_TURN
            self.context.last_speaker = "morty"
        elif self.context.last_speaker == "morty":
            self.state = PodcastState.RICK_TURN
            self.context.last_speaker = "rick"
        else:
            # First turn - Morty asks, Rick answers
            self.state = PodcastState.MORTY_TURN
            self.context.last_speaker = "morty"
        
        return self.state
    
    def handle_user_interrupt(self, user_input: str, source: str = "text"):
        """
        Handle a user interruption.
        
        Args:
            user_input: The user's message or detected sign
            source: "text" or "sign"
        """
        if source == "sign":
            self.context.pending_sign_input = user_input
        else:
            self.context.pending_user_input = user_input
        
        self.state = PodcastState.USER_INTERRUPT
    
    def resume_discussion(self) -> PodcastState:
        """Resume discussion after handling interrupt."""
        self.context.pending_user_input = None
        self.context.pending_sign_input = None
        self.state = PodcastState.DISCUSSING
        # DO NOT call next_turn() here - it will be called by _handle_discussion_turn
        # Calling it here causes double-increment of turn_count
        return self.state
    
    def pause(self):
        """Pause the podcast."""
        self._previous_state = self.state
        self.state = PodcastState.PAUSED
    
    def resume(self):
        """Resume from pause."""
        if hasattr(self, '_previous_state'):
            self.state = self._previous_state
        else:
            self.state = PodcastState.DISCUSSING
    
    def wrap_up(self):
        """Start wrapping up the podcast."""
        self.state = PodcastState.WRAP_UP
    
    def end(self):
        """End the podcast and return to idle."""
        self.state = PodcastState.IDLE
    
    def get_next_speaker(self) -> str:
        """Get who should speak next."""
        if self.state == PodcastState.MORTY_TURN:
            return "morty"
        elif self.state == PodcastState.RICK_TURN:
            return "rick"
        elif self.state == PodcastState.SUMMER_TURN:
            return "summer"
        elif self.state == PodcastState.USER_INTERRUPT:
            return "summer"  # Summer announces interrupts
        elif self.state in [PodcastState.WELCOME, PodcastState.WRAP_UP, PodcastState.NEW_TOPIC]:
            return "summer"  # Summer handles transitions
        return ""
    
    def should_continue(self) -> bool:
        """Check if podcast should continue."""
        return self.state not in [PodcastState.IDLE, PodcastState.WRAP_UP]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for API responses."""
        return {
            "state": self.state.name,
            "current_topic": self.context.current_topic,
            "turn_count": self.context.turn_count,
            "last_speaker": self.context.last_speaker,
            "topics_discussed": self.context.topics_discussed,
            "has_pending_input": bool(
                self.context.pending_user_input or self.context.pending_sign_input
            ),
        }

