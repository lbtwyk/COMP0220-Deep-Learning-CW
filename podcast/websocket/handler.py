"""
WebSocket Handler for Podcast

Manages WebSocket connections and routes messages between
the frontend and the podcast agents.
"""

import json
import asyncio
from typing import Optional, Dict, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect


class MessageType(Enum):
    """Types of WebSocket messages."""
    # Client → Server
    START = "start"            # Start podcast with topic
    INTERRUPT = "interrupt"    # User text interruption
    FRAME = "frame"           # Webcam frame (base64)
    PAUSE = "pause"           # Pause podcast
    RESUME = "resume"         # Resume podcast
    SKIP = "skip"             # Skip to next subtopic
    END = "end"               # End podcast
    SET_PERSONALITY = "set_personality"  # Switch personality mode
    SET_MODEL = "set_model"    # Switch between local/API model
    
    # Server → Client
    SPEECH = "speech"          # Agent speaking
    SIGN_DETECTED = "sign_detected"  # Sign language detected
    STATE = "state"            # State change notification
    REQUEST_TOPIC = "request_topic"  # Ask client for topic
    ENDED = "ended"            # Podcast ended
    ERROR = "error"            # Error message
    AGENTS_INFO = "agents_info"  # Agent information update


@dataclass
class WebSocketMessage:
    """Parsed WebSocket message."""
    type: MessageType
    data: Dict[str, Any]
    
    @classmethod
    def from_json(cls, raw: str) -> "WebSocketMessage":
        """Parse a JSON message."""
        parsed = json.loads(raw)
        msg_type = MessageType(parsed.get("type", ""))
        return cls(type=msg_type, data=parsed)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.value,
            **self.data,
        })


class PodcastWebSocketHandler:
    """
    Handles WebSocket connections for the podcast.
    
    This class manages:
    - Connection lifecycle
    - Message routing
    - Broadcasting to connected clients
    """
    
    def __init__(self):
        """Initialize the handler."""
        self.active_connections: Set[WebSocket] = set()
        self._message_handlers: Dict[MessageType, Callable] = {}
    
    def register_handler(self, msg_type: MessageType, handler: Callable):
        """Register a handler for a message type."""
        self._message_handlers[msg_type] = handler
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        self.active_connections.discard(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, websocket: WebSocket, msg_type: MessageType, data: Dict[str, Any]):
        """Send a message to a specific client."""
        message = WebSocketMessage(type=msg_type, data=data)
        try:
            json_msg = message.to_json()
            print(f"📨 Sending {msg_type.value} message ({len(json_msg)} bytes)")
            await websocket.send_text(json_msg)
            print(f"✅ Message sent successfully")
        except Exception as e:
            print(f"❌ Error sending message: {e}")
            import traceback
            traceback.print_exc()
            self.disconnect(websocket)
    
    async def broadcast(self, msg_type: MessageType, data: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        message = WebSocketMessage(type=msg_type, data=data)
        json_msg = message.to_json()
        
        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json_msg)
            except Exception:
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        self.active_connections -= disconnected
    
    async def handle_message(self, websocket: WebSocket, raw_message: str) -> Optional[Dict[str, Any]]:
        """
        Parse and route an incoming message.
        
        Args:
            websocket: The WebSocket that sent the message
            raw_message: Raw JSON message string
            
        Returns:
            Response data if any
        """
        try:
            message = WebSocketMessage.from_json(raw_message)
            
            # Check for registered handler
            handler = self._message_handlers.get(message.type)
            if handler:
                return await handler(websocket, message.data)
            
            print(f"No handler for message type: {message.type}")
            return None
            
        except json.JSONDecodeError as e:
            await self.send_message(websocket, MessageType.ERROR, {
                "error": f"Invalid JSON: {e}"
            })
            return None
        except ValueError as e:
            await self.send_message(websocket, MessageType.ERROR, {
                "error": f"Unknown message type: {e}"
            })
            return None
    
    async def listen(self, websocket: WebSocket):
        """
        Listen for messages on a WebSocket connection.
        
        This is the main loop for handling a client connection.
        """
        try:
            while True:
                raw_message = await websocket.receive_text()
                await self.handle_message(websocket, raw_message)
        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            self.disconnect(websocket)
    
    # ==================== Message Sending Helpers ====================
    
    async def send_speech(
        self,
        websocket: WebSocket,
        agent: str,
        text: str,
        audio_url: Optional[str] = None,
        model_source: str = "cloud"
    ):
        """Send a speech message from an agent."""
        await self.send_message(websocket, MessageType.SPEECH, {
            "agent": agent,
            "text": text,
            "audio_url": audio_url,
            "model": model_source,
        })
    
    async def send_state_update(self, websocket: WebSocket, state: str, topic: str = ""):
        """Send a state change notification."""
        await self.send_message(websocket, MessageType.STATE, {
            "state": state,
            "topic": topic,
        })
    
    async def send_sign_detected(
        self,
        websocket: WebSocket,
        sign: str,
        confidence: float,
    ):
        """Send a detected sign notification."""
        await self.send_message(websocket, MessageType.SIGN_DETECTED, {
            "sign": sign,
            "confidence": confidence,
        })
    
    async def request_topic(self, websocket: WebSocket):
        """Request a topic from the client."""
        await self.send_message(websocket, MessageType.REQUEST_TOPIC, {})
    
    async def send_ended(self, websocket: WebSocket, summary: str = ""):
        """Send podcast ended notification."""
        await self.send_message(websocket, MessageType.ENDED, {
            "summary": summary,
        })
    
    async def send_agents_info(
        self,
        websocket: WebSocket,
        agents: Dict[str, Any],
        personality: str,
    ):
        """Send agent information to client."""
        await self.send_message(websocket, MessageType.AGENTS_INFO, {
            "agents": agents,
            "personality": personality,
        })

