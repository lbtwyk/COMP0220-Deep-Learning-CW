"""
Base Agent class for the podcast system.

Provides common functionality for all agents including:
- LLM integration (OpenAI API or local Qwen3 model)
- Personality mode switching (fun vs professional)
- Message generation with context
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import os

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Global reference to shared local model (loaded once, reused by all agents)
_local_model = None
_local_tokenizer = None
_local_device = None

# Global reference to shared LSTM model
_lstm_model = None
_lstm_vocab = None
_lstm_config = None


def set_local_model(model, tokenizer, device):
    """Set the shared local model for all agents."""
    global _local_model, _local_tokenizer, _local_device
    _local_model = model
    _local_tokenizer = tokenizer
    _local_device = device


def get_local_model() -> Tuple[Any, Any, str]:
    """Get the shared local model."""
    return _local_model, _local_tokenizer, _local_device


def is_local_model_available() -> bool:
    """Check if local model is loaded."""
    return _local_model is not None


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
        model_type: str = "api",  # 'api', 'local', or 'lstm'
    ):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
            personality: Personality mode (fun or professional)
            model: OpenAI model to use
            temperature: Sampling temperature for responses
            model_type: Which model to use - 'api' (OpenAI), 'local' (Qwen3), or 'lstm'
        """
        self.config = config
        self.personality = personality
        self.model = model
        self.temperature = temperature
        self.model_type = model_type
        
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
    
    def set_model_type(self, model_type: str):
        """Switch between api, local, and lstm model."""
        self.model_type = model_type
    
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
        print(f"  [{self.name}] generate_response called (model_type={self.model_type})")
        
        # Use LSTM model if specified
        if self.model_type == "lstm":
            return await self._generate_lstm_response(
                conversation_history, current_topic, additional_context
            )
        
        # Use local Qwen3 model if specified
        if self.model_type == "local":
            return await self._generate_local_response(
                conversation_history, current_topic, additional_context
            )
        
        # Default: use OpenAI API
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
    
    async def _generate_local_response(
        self,
        conversation_history: List[Message],
        current_topic: str,
        additional_context: Optional[str] = None,
    ) -> str:
        """Generate a response using the local Qwen3 model."""
        model, tokenizer, device = get_local_model()
        
        if model is None:
            print(f"  [{self.name}] ❌ Local model not loaded, using fallback")
            return self._generate_fallback_response(current_topic)
        
        print(f"  [{self.name}] Using local Qwen3 model on {device}")
        
        # Build system prompt with topic context
        system_content = self.system_prompt
        system_content += f"\n\nCurrent topic: {current_topic}"
        if additional_context:
            system_content += f"\n\nAdditional context: {additional_context}"
        
        # Add anti-hallucination constraints for local model
        system_content += """

CRITICAL RULES:
- Do NOT invent personal anecdotes or stories (e.g., "my cousin who's Deaf...")
- Base responses on established ASL and Deaf culture knowledge only
- If you don't know something, say so honestly
- Stick to verifiable facts about sign language
- Keep responses conversational but grounded in reality"""
        
        # Build conversational prompt from history
        # Format history with speaker attribution for better context
        history_text = ""
        for msg in conversation_history[-10:]:
            if isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = msg.content
            if content:
                history_text += content + "\n\n"
        
        # SOLUTION 1: Conversational prompt engineering for local model
        # Build a dialogue-oriented prompt that encourages back-and-forth
        co_host = "Taylor" if self.name.lower() in ["dave", "rick"] else "Dave"
        
        if history_text:
            user_prompt = f"""You are having a podcast conversation with your co-host {co_host} about {current_topic}.

Previous conversation:
{history_text}

Now it's your turn to speak. Respond naturally as if you're having a real conversation with {co_host}. You can:
- Build on what was just said
- Ask {co_host} a follow-up question to keep the discussion going
- Share an interesting related point and invite their perspective

Keep it conversational and engaging (2-4 sentences). Remember: this is a DIALOGUE, not a lecture."""
        else:
            user_prompt = f"""You are starting a podcast conversation with your co-host {co_host} about {current_topic}.
            
Introduce the topic in a conversational way and ask {co_host} a question to get the discussion going. Keep it natural and engaging (2-4 sentences)."""
        
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            import random
            
            # Import local inference function
            from inference import generate_response as local_generate
            
            # Prepare stop strings
            # IMPORTANT: Do NOT include self.name in stop strings
            all_stops = [
                "Rick:", "Morty:", "Summer:", 
                "Dave:", "Taylor:", "Pat:",
                "User:", "System:",
            ]
            # Filter out current agent's name from stop strings
            stop_strings = [s for s in all_stops if not s.lower().startswith(self.name.lower())]
            
            def _run_local_inference():
                return local_generate(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    user_prompt=user_prompt,
                    system_prompt=system_content,
                    max_new_tokens=300,
                    temperature=self.temperature,
                    stop_strings=stop_strings,
                )
            
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                # Run in thread to not block event loop
                response = await asyncio.wait_for(
                    loop.run_in_executor(executor, _run_local_inference),
                    timeout=60.0
                )
            finally:
                executor.shutdown(wait=False)
            
            # Clean up response
            cleaned_response = response.strip()
            # Regex to remove "Name:" or "[Name]:" at start
            import re
            name_pattern = re.compile(f"^[\\[\\(]?{self.name}[\\]\\)]?:\\s*", re.IGNORECASE)
            cleaned_response = name_pattern.sub("", cleaned_response)
            
            # SOLUTION 3: Add conversational hooks if response doesn't engage
            # Check if response naturally engages the co-host
            engages_cohost = (
                '?' in cleaned_response or  # Has a question
                co_host.lower() in cleaned_response.lower() or  # Mentions co-host
                any(phrase in cleaned_response.lower() for phrase in [
                    'what do you think', 'have you', 'would you', 'do you',
                    'your thoughts', 'your perspective'
                ])
            )
            
            if not engages_cohost:
                # Add a conversational bridge
                bridges = [
                    f" What do you think, {co_host}?",
                    f" {co_host}, have you noticed this?",
                    f" Does that resonate with you, {co_host}?",
                    f" {co_host}, what's your take on that?",
                ]
                cleaned_response += random.choice(bridges)
                print(f"  [{self.name}] 🔗 Added conversational hook")
            
            print(f"  [{self.name}] ✅ [LOCAL] Response ({len(cleaned_response)} chars): {cleaned_response[:80]}...")
            return cleaned_response
            

            
        except asyncio.TimeoutError:
            print(f"  [{self.name}] ⏱️ Local model timeout after 60s")
            return self._generate_fallback_response(current_topic)
        except Exception as e:
            print(f"  [{self.name}] ❌ Local model error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_response(current_topic)
    
    async def _generate_lstm_response(
        self,
        conversation_history: List[Message],
        current_topic: str,
        additional_context: Optional[str] = None,
    ) -> str:
        """Generate a response using the LSTM baseline model."""
        global _lstm_model, _lstm_vocab, _lstm_config
        
        # Load LSTM model if not already loaded
        if _lstm_model is None:
            try:
                from pathlib import Path
                from inference_lstm import load_model as load_lstm_model
                
                checkpoint_path = Path("./lstm_baseline/checkpoint_best.pt")
                if not checkpoint_path.exists():
                    print(f"  [{self.name}] ❌ LSTM checkpoint not found")
                    return self._generate_fallback_response(current_topic)
                
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _lstm_model, _lstm_vocab, _lstm_config = load_lstm_model(
                    checkpoint_path=checkpoint_path,
                    device=device
                )
                print(f"  [{self.name}] ✅ LSTM model loaded on {device}")
            except Exception as e:
                print(f"  [{self.name}] ❌ Failed to load LSTM model: {e}")
                return self._generate_fallback_response(current_topic)
        
        print(f"  [{self.name}] Using LSTM model")
        
        # Build a simple question from the topic and context
        if additional_context:
            question = f"{additional_context} about {current_topic}"
        else:
            question = f"Tell me about {current_topic}"
        
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            from inference_lstm import generate_response as lstm_generate
            
            def _run_lstm_inference():
                return lstm_generate(
                    model=_lstm_model,
                    vocab=_lstm_vocab,
                    question=question,
                    config=_lstm_config,
                    temperature=self.temperature
                )
            
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(executor, _run_lstm_inference),
                    timeout=30.0
                )
            finally:
                executor.shutdown(wait=False)
            
            cleaned_response = response.strip()
            print(f"  [{self.name}] ✅ [LSTM] Response ({len(cleaned_response)} chars): {cleaned_response[:80]}...")
            return cleaned_response
            
        except asyncio.TimeoutError:
            print(f"  [{self.name}] ⏱️ LSTM model timeout after 30s")
            return self._generate_fallback_response(current_topic)
        except Exception as e:
            print(f"  [{self.name}] ❌ LSTM model error: {type(e).__name__}: {e}")
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

