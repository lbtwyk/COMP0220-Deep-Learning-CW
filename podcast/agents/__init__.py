"""
Podcast Agents Module

Contains the three agents:
- Rick: The genius expert (or "Dr. Alex" in professional mode)
- Morty: The curious host (or "Morgan" in professional mode)  
- Summer: The coordinator/producer (or "Sam" in professional mode)
"""

from .base import BaseAgent, AgentPersonality
from .rick import RickAgent
from .morty import MortyAgent
from .summer import SummerAgent

__all__ = [
    "BaseAgent",
    "AgentPersonality",
    "RickAgent",
    "MortyAgent",
    "SummerAgent",
]

