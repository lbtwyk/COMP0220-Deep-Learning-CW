"""
Summer Agent - The Coordinator/Producer

In FUN mode: Summer Smith style - slightly exasperated but helpful producer
In PROFESSIONAL mode: Sam - efficient, friendly podcast producer
"""

from typing import Optional, List, Dict, Any
from .base import BaseAgent, AgentConfig, AgentPersonality, Message


# Summer's configuration
SUMMER_CONFIG = AgentConfig(
    name="Coordinator",
    role="The podcast producer who keeps things on track",
    fun_name="Summer",
    professional_name="Pat",
    emoji="🎯",
    
    # ElevenLabs voice IDs
    fun_voice_id="MF3mGyEYCl7XYWbV9V6O",        # Elli (young, energetic)
    professional_voice_id="EXAVITQu4vr4xnSDxMaL",  # Bella (soft, friendly)
    
    # Fun mode: Summer Smith personality
    fun_system_prompt="""You are Summer, the producer of a podcast about sign language and Deaf culture hosted by Rick and Morty. You're behind the scenes but occasionally interject.

Your personality:
- Slightly exasperated but ultimately helpful
- Keep things moving and on-topic
- Brief and to-the-point (1 sentence usually)
- Occasionally roast Rick and Morty for going off-track
- Announce user interactions and topic changes

Example style:
"Ugh, okay you two, the user wants to know about fingerspelling. Try to stay on topic this time."
"Hold up—we have a question from the audience."
"Rick, that's actually interesting for once."

You only speak when needed: announcing topics, handling interruptions, keeping things on track.""",
    
    # Professional mode: Pat (Foo Fighters) personality
    professional_system_prompt="""You are Pat, the producer of an educational podcast about sign language and Deaf culture hosted by Dave and Taylor.

Your personality:
- Efficient and friendly
- Keep the podcast flowing smoothly
- Brief announcements (1 sentence)
- Introduce topics and handle transitions
- Announce user questions and interactions

Example style:
"Great question from our audience about fingerspelling."
"Let's move on to our next topic: Deaf culture."
"We have a viewer question coming in."

You only speak when needed for transitions and announcements."""
)


class SummerAgent(BaseAgent):
    """
    The Coordinator/Producer Agent.
    
    Fun mode: Summer - exasperated but helpful producer
    Professional mode: Pat - efficient, friendly coordinator
    
    Unlike Rick and Morty, Summer handles:
    - Topic introductions and transitions
    - User interrupt announcements
    - Keeping the podcast on track
    - Sign language recognition announcements
    """
    
    def __init__(
        self,
        personality: AgentPersonality = AgentPersonality.PROFESSIONAL,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,  # Lower temperature for more consistent coordination
        use_local_model: bool = False,
    ):
        super().__init__(
            config=SUMMER_CONFIG,
            personality=personality,
            model=model,
            temperature=temperature,
            use_local_model=use_local_model,
        )
    
    def _generate_fallback_response(self, topic: str) -> str:
        """Generate fallback when API unavailable."""
        if self.personality == AgentPersonality.FUN:
            return f"Okay, we're talking about {topic} now. Try to focus, you two."
        return f"Let's discuss {topic}."
    
    # ==================== Coordinator-specific methods ====================
    
    def announce_topic(self, topic: str) -> str:
        """Announce a new topic."""
        if self.personality == AgentPersonality.FUN:
            return f"Alright, the user wants to learn about {topic}. Rick, Morty—try not to go on too many tangents this time."
        return f"Today's topic is {topic}. Let's dive in!"
    
    def announce_user_interrupt(self, user_input: str, source: str = "text") -> str:
        """Announce a user interruption."""
        if source == "sign":
            source_text = "signed" if self.personality == AgentPersonality.FUN else "via sign language"
        else:
            source_text = "typed" if self.personality == AgentPersonality.FUN else "sent"
        
        if self.personality == AgentPersonality.FUN:
            return f"Hold up—the user just {source_text}: \"{user_input}\""
        return f"We have a question from our audience: \"{user_input}\""
    
    def announce_sign_detected(self, sign: str, confidence: float) -> str:
        """Announce a detected sign from webcam."""
        if self.personality == AgentPersonality.FUN:
            if confidence > 0.9:
                return f"Whoa, nice signing! The user just signed '{sign}'."
            return f"I think the user signed '{sign}'... {int(confidence*100)}% sure."
        else:
            return f"The viewer signed: '{sign}'."
    
    def announce_topic_transition(self, old_topic: str, new_topic: str) -> str:
        """Announce transitioning to a new topic."""
        if self.personality == AgentPersonality.FUN:
            return f"Okay, we've covered {old_topic}. Moving on to {new_topic}—and Rick, keep it under 10 minutes this time."
        return f"Great discussion on {old_topic}. Let's move to {new_topic}."
    
    def announce_wrap_up(self) -> str:
        """Announce the podcast is wrapping up."""
        if self.personality == AgentPersonality.FUN:
            return "Alright, that's all the time we have. Rick, Morty—say goodbye."
        return "Thank you for joining us today! We hope you learned something new."
    
    def get_redirect_message(self, current_topic: str) -> str:
        """Get a message to redirect back to topic."""
        if self.personality == AgentPersonality.FUN:
            return f"Ugh, can we get back to {current_topic}? We're supposed to be educational."
        return f"Let's refocus on {current_topic}."
    
    def welcome_message(self) -> str:
        """Get the welcome message for new users."""
        if self.personality == AgentPersonality.FUN:
            return "Hey! Welcome to the SignTutor podcast. What topic about sign language or Deaf culture do you want Rick and Morty to discuss?"
        return "Welcome to SignTutor! What topic about sign language or Deaf culture would you like Dave and Taylor to discuss today?"

