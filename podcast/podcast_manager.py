"""
Podcast Manager - Main orchestrator for the agentic podcast.
"""

import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass

from fastapi import WebSocket

from .agents import RickAgent, MortyAgent, SummerAgent, AgentPersonality
from .services import PodcastStateMachine, PodcastState, ConversationMemory
from .websocket import PodcastWebSocketHandler
from .websocket.handler import MessageType


@dataclass
class PodcastConfig:
    """Configuration for a podcast session."""
    personality: AgentPersonality = AgentPersonality.PROFESSIONAL
    tts_provider: str = "browser"  # Use browser TTS by default
    turn_delay_ms: int = 1500
    use_local_model: bool = False  # Use local Qwen3 model instead of OpenAI API


class PodcastManager:
    """Main podcast orchestrator."""
    
    def __init__(self, config: Optional[PodcastConfig] = None):
        self.config = config or PodcastConfig()
        
        # Initialize agents with local model setting
        self.rick = RickAgent(personality=self.config.personality, use_local_model=self.config.use_local_model)
        self.morty = MortyAgent(personality=self.config.personality, use_local_model=self.config.use_local_model)
        self.summer = SummerAgent(personality=self.config.personality, use_local_model=self.config.use_local_model)
        
        # Initialize services
        self.state_machine = PodcastStateMachine()
        self.memory = ConversationMemory()
        self.ws_handler = PodcastWebSocketHandler()
        
        self._current_ws: Optional[WebSocket] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        self._register_handlers()
    
    def _register_handlers(self):
        """Register WebSocket message handlers."""
        self.ws_handler.register_handler(MessageType.START, self._handle_start)
        self.ws_handler.register_handler(MessageType.INTERRUPT, self._handle_interrupt)
        self.ws_handler.register_handler(MessageType.PAUSE, self._handle_pause)
        self.ws_handler.register_handler(MessageType.RESUME, self._handle_resume)
        self.ws_handler.register_handler(MessageType.SKIP, self._handle_skip)
        self.ws_handler.register_handler(MessageType.END, self._handle_end)
        self.ws_handler.register_handler(MessageType.SET_PERSONALITY, self._handle_set_personality)
        self.ws_handler.register_handler(MessageType.SET_MODEL, self._handle_set_model)
    
    def set_use_local_model(self, use_local: bool):
        """Switch model source for all agents."""
        self.config.use_local_model = use_local
        self.rick.set_use_local_model(use_local)
        self.morty.set_use_local_model(use_local)
        self.summer.set_use_local_model(use_local)
    
    def set_personality(self, personality: AgentPersonality):
        """Switch personality mode for all agents."""
        self.config.personality = personality
        self.rick.set_personality(personality)
        self.morty.set_personality(personality)
        self.summer.set_personality(personality)
    
    def get_agents_info(self) -> Dict[str, Any]:
        """Get info about all agents."""
        return {
            "rick": self.rick.to_dict(),
            "morty": self.morty.to_dict(),
            "summer": self.summer.to_dict(),
        }
    
    async def run_session(self, websocket: WebSocket):
        """Run a podcast session."""
        self._current_ws = websocket
        self._running = True
        
        try:
            await self.ws_handler.connect(websocket)
            
            # Send agent info
            await self.ws_handler.send_agents_info(
                websocket, self.get_agents_info(), self.config.personality.value
            )
            
            # Welcome
            self.state_machine.start()
            welcome = self.summer.welcome_message()
            await self._send_speech("summer", welcome)
            await self.ws_handler.request_topic(websocket)
            
            # Listen for messages
            await self.ws_handler.listen(websocket)
            
        except Exception as e:
            print(f"Podcast error: {e}")
        finally:
            self._running = False
            self._current_ws = None
    
    async def _send_speech(self, agent: str, text: str):
        """Send speech from an agent."""
        if not self._current_ws:
            print(f"⚠️ No WebSocket connection, cannot send speech from {agent}")
            return
        
        if not text or not text.strip():
            print(f"⚠️ Empty text from {agent}, skipping")
            return
        
        print(f"📤 Sending speech from {agent} ({len(text)} chars)")
        self.memory.add_turn(agent, text)
        
        # Determine model source
        model_source = "local" if self.config.use_local_model else "cloud"
        
        try:
            await self.ws_handler.send_speech(
                self._current_ws, 
                agent, 
                text, 
                None,
                model_source=model_source
            )
            print(f"✅ Speech sent successfully from {agent} ({model_source})")
        except Exception as e:
            print(f"❌ Error sending speech from {agent}: {e}")
            import traceback
            traceback.print_exc()
    
    async def _run_conversation_loop(self):
        """Main conversation loop - runs until max turns or podcast ends."""
        print("Conversation loop started")
        
        while self._running and self.state_machine.should_continue():
            state = self.state_machine.state
            
            print(f"Loop iteration, state: {state.name}, turn: {self.state_machine.context.turn_count}")
            
            if state == PodcastState.PAUSED:
                await asyncio.sleep(1)
                continue
            
            if state == PodcastState.USER_INTERRUPT:
                print("Handling interrupt...")
                await self._handle_interrupt_turn()

                await asyncio.sleep(2)  # Wait after interrupt
                continue
            
            if state == PodcastState.NEW_TOPIC:
                print("New topic requested, waiting for input...")
                await asyncio.sleep(1)
                continue
            
            if state in [PodcastState.DISCUSSING, PodcastState.MORTY_TURN, PodcastState.RICK_TURN]:
                print(f"Handling discussion turn, current speaker: {self.state_machine.context.last_speaker}")
                await self._handle_discussion_turn()
                # Wait longer between turns
                await asyncio.sleep(self.config.turn_delay_ms / 1000)
            else:
                # Unknown state, wait a bit
                await asyncio.sleep(1)
        
        print(f"Conversation loop ended. Running: {self._running}")
    
    async def _handle_discussion_turn(self):
        """Handle one turn of discussion."""
        try:
            topic = self.state_machine.context.current_topic
            if not topic:
                print("No topic set, transitioning to NEW_TOPIC")
                self.state_machine.state = PodcastState.NEW_TOPIC
                # Optionally request topic from frontend
                # await self.ws_handler.request_topic(self._current_ws)
                return
            
            # Get next speaker BEFORE generating
            next_state = self.state_machine.next_turn()
            speaker = self.state_machine.get_next_speaker()
            
            print(f"Next speaker: {speaker}, state: {next_state.name}")
            
            if not speaker:
                print("No speaker determined, skipping")
                return
            
            # Generate response
            print(f"Generating response for {speaker}...")
            try:
                if speaker == "morty":
                    response = await self.morty.generate_response(
                        self.memory.get_context(), 
                        topic
                    )
                    
                    # Check for interrupt during generation
                    if self.state_machine.state == PodcastState.USER_INTERRUPT:
                        print("⛔ Interrupt received during Morty's generation, discarding response")
                        return

                    print(f"✅ Morty response generated: {response[:50]}...")
                    await self._send_speech("morty", response)
                    print(f"✅ Morty speech sent to frontend")
                elif speaker == "rick":
                    response = await self.rick.generate_response(
                        self.memory.get_context(), 
                        topic
                    )
                    
                    # Check for interrupt during generation
                    if self.state_machine.state == PodcastState.USER_INTERRUPT:
                        print("⛔ Interrupt received during Rick's generation, discarding response")
                        return

                    print(f"✅ Rick response generated: {response[:50]}...")
                    await self._send_speech("rick", response)
                    print(f"✅ Rick speech sent to frontend")
                else:
                    print(f"Unknown speaker: {speaker}")
            except Exception as e:
                print(f"❌ Error in discussion turn for {speaker}: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error in discussion turn: {e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_interrupt_turn(self):
        """Handle user interrupt."""
        try:
            ctx = self.state_machine.context
            user_input = ctx.pending_user_input or ctx.pending_sign_input or ""
            
            if not user_input:
                print("No user input in interrupt, resuming")
                self.state_machine.resume_discussion()
                return
            
            print(f"Handling interrupt: {user_input}")
            
            # If no topic is set, use the interrupt as the topic
            if not self.state_machine.context.current_topic:
                print(f"No topic set, using interrupt as topic: {user_input}")
                self.state_machine.set_topic(user_input)
            
            announcement = self.summer.announce_user_interrupt(user_input, "text")
            await self._send_speech("summer", announcement)
            await asyncio.sleep(1)
            
            self.memory.add_turn("user", user_input)
            
            response = await self.rick.generate_response(
                self.memory.get_context(), 
                self.state_machine.context.current_topic,
                additional_context=f"User asked: {user_input}"
            )
            await self._send_speech("rick", response)
            
            # Resume and set last speaker
            self.state_machine.context.last_speaker = "rick"
            self.state_machine.context.turn_count = 0  # Reset topic turn count on interrupt
            self.state_machine.resume_discussion()
            
        except Exception as e:
            print(f"Error handling interrupt: {e}")
            import traceback
            traceback.print_exc()
            self.state_machine.resume_discussion()
    
    # Message handlers
    async def _handle_start(self, websocket: WebSocket, data: Dict[str, Any]):
        topic = data.get("topic", "")
        if not topic:
            await self.ws_handler.request_topic(websocket)
            return
        
        print(f"Starting podcast with topic: {topic}")
        
        self.state_machine.set_topic(topic)
        self.memory.add_topic(topic)
        
        # Summer intro
        await self._send_speech("summer", self.summer.announce_topic(topic))
        await asyncio.sleep(1)  # Small delay
        
        # Morty asks
        await self._send_speech("morty", self.morty.get_intro_message(topic))
        await asyncio.sleep(1)
        
        # Rick explains
        print("Generating Rick's initial response...")
        response = await self.rick.generate_response(self.memory.get_context(), topic)
        await self._send_speech("rick", response)
        
        # Set state to discussing and initialize turn counter
        self.state_machine.context.last_speaker = "rick"  # Rick just spoke
        self.state_machine.state = PodcastState.DISCUSSING
        
        # Wait a bit before starting the loop
        await asyncio.sleep(3)
        
        # Start loop
        if self._task is None or self._task.done():
            print("Starting conversation loop task...")
            self._task = asyncio.create_task(self._run_conversation_loop())
        else:
            print("Loop task already running")
        
        await self.ws_handler.send_state_update(websocket, "discussing", topic)
    
    async def _handle_interrupt(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle user interrupt - restart loop if needed."""
        message = data.get("message", "")
        if message:
            self.state_machine.handle_user_interrupt(message, "text")
            
            # Restart loop if it has stopped
            if self._task is None or self._task.done():
                print("Restarting conversation loop for interrupt...")
                self._task = asyncio.create_task(self._run_conversation_loop())
    
    async def _handle_pause(self, websocket: WebSocket, data: Dict[str, Any]):
        self.state_machine.pause()
        await self.ws_handler.send_state_update(websocket, "paused")
    
    async def _handle_resume(self, websocket: WebSocket, data: Dict[str, Any]):
        self.state_machine.resume()
        await self.ws_handler.send_state_update(websocket, "discussing", self.state_machine.context.current_topic)
    
    async def _handle_skip(self, websocket: WebSocket, data: Dict[str, Any]):
        self.state_machine.state = PodcastState.NEW_TOPIC
        await self.ws_handler.request_topic(websocket)
    
    async def _handle_end(self, websocket: WebSocket, data: Dict[str, Any]):
        self._running = False
        await self._send_speech("summer", self.summer.announce_wrap_up())
        await self.ws_handler.send_ended(websocket, "")
    
    async def _handle_set_personality(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle personality switch during active podcast."""
        new_personality_str = data.get("personality", "fun")
        new_personality = AgentPersonality(new_personality_str)
        old_personality = self.config.personality
        
        # Only switch if different
        if new_personality == old_personality:
            return
        
        print(f"Switching personality from {old_personality.value} to {new_personality_str}")
        
        # Update all agents
        self.set_personality(new_personality)
        
        # Send agent info update
        await self.ws_handler.send_agents_info(websocket, self.get_agents_info(), new_personality_str)
        
        # Add a transition message from Summer
        if self.state_machine.state in [PodcastState.DISCUSSING, PodcastState.MORTY_TURN, PodcastState.RICK_TURN]:
            if new_personality == AgentPersonality.FUN:
                transition_msg = "Alright, switching to fun mode! Rick and Morty are back!"
            else:
                transition_msg = "Switching to professional mode. Dave and Taylor will continue the discussion."
            
            await self._send_speech("summer", transition_msg)
            await asyncio.sleep(1)  # Brief pause after transition
    
    async def _handle_set_model(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle model source switch during active podcast."""
        use_local = data.get("use_local", False)
        old_use_local = self.config.use_local_model
        
        # Only switch if different
        if use_local == old_use_local:
            return
        
        model_name = "Local (Qwen3-4B)" if use_local else "Cloud (OpenAI)"
        print(f"Switching model to: {model_name}")
        
        # Update all agents
        self.set_use_local_model(use_local)
        
        # Notify frontend of model change
        await self.ws_handler.send_message(websocket, MessageType.STATE, {
            "state": "model_changed",
            "use_local": use_local,
        })
