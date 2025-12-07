"""
Base Agent class for the podcast system.

Provides common functionality for all agents including:
- LLM integration (OpenAI API)
- Personality mode switching (fun vs professional)
- Message generation with context
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AgentPersonality(Enum):
    """Personality mode for agents."""
    FUN = "fun"              # Rick & Morty style - playful, character-driven
    PROFESSIONAL = "professional"  # Standard educational tone


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str                          # Agent display name
    role: str                          # Agent role description
    fun_name: str                      # Name in fun mode (e.g., "Rick")
    professional_name: str             # Name in professional mode (e.g., "Dr. Alex")
    fun_voice_id: str                  # ElevenLabs voice ID for fun mode
    professional_voice_id: str         # ElevenLabs voice ID for professional mode
    fun_system_prompt: str             # System prompt for fun personality
    professional_system_prompt: str    # System prompt for professional personality
    emoji: str = ""                    # Emoji for display


@dataclass
class Message:
    """A message in the conversation."""
    role: str           # "system", "user", "assistant"
    content: str        # Message content
    agent_name: str = ""  # Which agent said this (for multi-agent context)
    timestamp: float = 0.0


class BaseAgent(ABC):
    """
    Base class for all podcast agents.
    
    Provides:
    - Personality switching (fun/professional)
    - LLM-based response generation
    - Voice ID selection for TTS
    """
    
    def __init__(
        self,
        config: AgentConfig,
        personality: AgentPersonality = AgentPersonality.FUN,
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,
    ):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
            personality: Personality mode (fun or professional)
            model: OpenAI model to use
            temperature: Sampling temperature for responses
        """
        self.config = config
        self.personality = personality
        self.model = model
        self.temperature = temperature
        
        # Initialize OpenAI client
        self._client: Optional[OpenAI] = None
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    @property
    def name(self) -> str:
        """Get the agent's current display name based on personality."""
        if self.personality == AgentPersonality.FUN:
            return self.config.fun_name
        return self.config.professional_name
    
    @property
    def voice_id(self) -> str:
        """Get the ElevenLabs voice ID based on personality."""
        if self.personality == AgentPersonality.FUN:
            return self.config.fun_voice_id
        return self.config.professional_voice_id
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt based on personality."""
        if self.personality == AgentPersonality.FUN:
            return self.config.fun_system_prompt
        return self.config.professional_system_prompt
    
    @property
    def emoji(self) -> str:
        """Get the agent's emoji."""
        return self.config.emoji
    
    def set_personality(self, personality: AgentPersonality):
        """Switch personality mode."""
        self.personality = personality
    
    async def generate_response(
        self,
        conversation_history: List[Message],
        current_topic: str,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Generate a response based on conversation history.
        
        Args:
            conversation_history: List of previous messages
            current_topic: The current discussion topic
            additional_context: Optional extra context (e.g., user interruption)
            
        Returns:
            Generated response text
        """
        print(f"  [{self.name}] generate_response called")
        
        if not self._client:
            print(f"  [{self.name}] No OpenAI client, using fallback")
            return self._generate_fallback_response(current_topic)
        
        print(f"  [{self.name}] Building messages...")
        # Build messages for the API
        messages = self._build_messages(
            conversation_history,
            current_topic,
            additional_context
        )
        print(f"  [{self.name}] Built {len(messages)} messages")
        
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            import sys
            
            print(f"  [{self.name}] Calling OpenAI API (model: {self.model}, {len(messages)} messages)...")
            
            # Python 3.9+ has asyncio.to_thread
            print(f"  [{self.name}] Starting API call in thread...")
            if sys.version_info >= (3, 9):
                # Use asyncio.to_thread (Python 3.9+)
                print(f"  [{self.name}] Using asyncio.to_thread")
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._client.chat.completions.create,
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=300,
                        ),
                        timeout=30.0
                    )
                    print(f"  [{self.name}] API call completed")
                except AttributeError:
                    print(f"  [{self.name}] asyncio.to_thread not available, using executor")
                    # Fallback for older Python
                    loop = asyncio.get_event_loop()
                    executor = ThreadPoolExecutor(max_workers=1)
                    response = await asyncio.wait_for(
                        loop.run_in_executor(
                            executor,
                            lambda: self._client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=self.temperature,
                                max_tokens=300,
                            )
                        ),
                        timeout=30.0
                    )
                    executor.shutdown(wait=False)
            else:
                print(f"  [{self.name}] Using run_in_executor (Python < 3.9)")
                # Python 3.7-3.8
                loop = asyncio.get_event_loop()
                executor = ThreadPoolExecutor(max_workers=1)
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        lambda: self._client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=300,
                        )
                    ),
                    timeout=30.0
                )
                executor.shutdown(wait=False)
            
            result = response.choices[0].message.content.strip()
            print(f"  [{self.name}] ✅ Response ({len(result)} chars): {result[:80]}...")
            return result
            
        except asyncio.TimeoutError:
            print(f"  [{self.name}] ⏱️ Timeout after 30s")
            return self._generate_fallback_response(current_topic)
        except Exception as e:
            print(f"  [{self.name}] ❌ Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_response(current_topic)
    
    def _build_messages(
        self,
        conversation_history: List[Message],
        current_topic: str,
        additional_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the messages list for the OpenAI API."""
        messages = []
        
        # Add system prompt with topic context
        system_content = self.system_prompt
        system_content += f"\n\nCurrent topic: {current_topic}"
        if additional_context:
            system_content += f"\n\nAdditional context: {additional_context}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history (last N messages for context)
        for msg in conversation_history[-10:]:
            # Handle both Message objects and dicts
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Strip any existing agent tags like [RICK]: or [MORTY]:
                if content.startswith("[") and "]: " in content:
                    content = content.split("]: ", 1)[1] if "]: " in content else content
            else:
                role = msg.role
                content = msg.content
                # Strip any existing agent tags
                if content.startswith("[") and "]: " in content:
                    content = content.split("]: ", 1)[1] if "]: " in content else content
            
            messages.append({"role": role, "content": content})
        
        return messages
    
    @abstractmethod
    def _generate_fallback_response(self, topic: str) -> str:
        """Generate a fallback response when API is unavailable."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent info to dictionary for API responses."""
        return {
            "name": self.name,
            "role": self.config.role,
            "emoji": self.emoji,
            "voice_id": self.voice_id,
            "personality": self.personality.value,
        }

