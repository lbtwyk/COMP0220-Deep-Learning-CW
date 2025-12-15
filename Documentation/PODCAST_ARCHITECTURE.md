# 🎙️ SignTutor Agentic Podcast System

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (React)                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Webcam Feed    │  │  Podcast View   │  │  User Controls              │  │
│  │  (Sign Input)   │  │  (Agent Avatars)│  │  (Topic, Interrupt, Mute)   │  │
│  └────────┬────────┘  └────────▲────────┘  └──────────────┬──────────────┘  │
│           │                    │                          │                  │
│           └──────────┬─────────┴──────────────────────────┘                  │
│                      │ WebSocket (bidirectional)                             │
└──────────────────────┼───────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────────────────┐
│                        BACKEND (FastAPI + WebSocket)                          │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      COORDINATOR AGENT (Summer) 🎯                       │ │
│  │  • "Ugh, okay you two, stay on topic..."                                 │ │
│  │  • Prompts user for initial topic                                        │ │
│  │  • Orchestrates conversation flow                                        │ │
│  │  • Monitors webcam frames → Sign Language Recognition                   │ │
│  │  • Detects and handles user interruptions                                │ │
│  │  • Decides when to switch speakers / inject user input                   │ │
│  └─────────────────────────────────┬───────────────────────────────────────┘ │
│                                    │                                          │
│           ┌────────────────────────┼────────────────────────┐                │
│           ▼                        ▼                        ▼                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  AGENT: Morty   │    │  AGENT: Rick    │    │   Sign Language Model   │  │
│  │  (The Host) 😰  │◄──►│ (The Expert) 🥒 │    │   (Visual Recognition)  │  │
│  │                 │    │                 │    │                         │  │
│  │ "Oh geez, so    │    │ "Listen Morty,  │    │ • MediaPipe Hands       │  │
│  │ you're saying   │    │ *burp* ASL is   │    │ • Custom classifier     │  │
│  │ that..."        │    │ way more than   │    │ • Real-time inference   │  │
│  │                 │    │ hand waving"    │    │                         │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        SHARED SERVICES                                   │ │
│  │  • TTS Engine (ElevenLabs/Google) - Different voices per agent          │ │
│  │  • Conversation Memory (context window)                                  │ │
│  │  • Topic Queue & State Machine                                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Agent Personalities

### 🥒 Rick (The Genius Expert)
- **Voice**: Raspy, confident, occasionally burps mid-sentence
- **Role**: The brilliant (if chaotic) expert on sign language and Deaf culture
- **Personality**: 
  - Genius-level knowledge, delivers facts with sardonic wit
  - "Listen Morty, *burp* the 5 parameters of ASL aren't just random—they're the fundamental building blocks of visual language!"
  - Occasionally goes on tangents but always brings it back with profound insights
  - Uses scientific analogies and interdimensional references
- **ElevenLabs Voice**: `ErXwobaYiN019PkySvjV` (Rick Sanchez - Raspy Genius) ✅ Already configured!

### 😰 Morty (The Curious Host)
- **Voice**: Nervous, stammering, relatable
- **Role**: The audience surrogate who asks the questions we're all thinking
- **Personality**:
  - "Oh geez Rick, s-so you're saying Deaf culture is like... a whole separate thing from just not hearing?"
  - Genuinely curious, sometimes confused, always learning
  - Occasionally has surprising insights that impress even Rick
  - Makes complex topics accessible through his questions
- **ElevenLabs Voice**: `yoZ06aMxZJJ28mfd3POQ` (Sam - nervous, youthful) or custom

### 🎯 Summer (The Coordinator)
- **Voice**: Brief, slightly exasperated but helpful
- **Role**: Behind-the-scenes orchestration (like a podcast producer)
- **Personality**:
  - "Ugh, okay you two, the user wants to know about fingerspelling. Try to stay on topic this time."
  - Keeps the podcast moving, handles interruptions
  - Occasionally roasts Rick and Morty for going off-track
- **Responsibilities**:
  1. Prompt user for initial topic
  2. Parse webcam frames for sign language input
  3. Detect user interruptions (raised hand, specific signs)
  4. Inject user questions into the conversation
  5. Keep track of time and topic progression
- **ElevenLabs Voice**: `MF3mGyEYCl7XYWbV9V6O` (Elli - Young & Energetic)

## Conversation Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    STATE MACHINE                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  [IDLE] ──(user joins)──► [WELCOME]                          │
│                               │                               │
│                        (ask for topic)                        │
│                               ▼                               │
│                        [TOPIC_INPUT] ◄─────────┐              │
│                               │                │              │
│                        (topic received)        │              │
│                               ▼                │              │
│                        [DISCUSSING] ◄──────────┤              │
│                          │      │              │              │
│              (natural)   │      │  (interrupt) │              │
│                 ▼        │      ▼              │              │
│          [MORTY_TURN]    │  [USER_INTERRUPT]   │              │
│                 │        │      │              │              │
│                 ▼        │      │              │              │
│          [RICK_TURN] ────┘      │              │              │
│                                 │              │              │
│                          (resume)              │              │
│                                 └──────────────┘              │
│                                                               │
│  [DISCUSSING] ──(topic exhausted)──► [NEW_TOPIC?]            │
│                                           │                   │
│                              (yes)        │ (no)              │
│                                ▼          ▼                   │
│                         [TOPIC_INPUT]  [WRAP_UP] ──► [IDLE]  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## WebSocket Message Protocol

### Client → Server
```json
// Start podcast with topic
{"type": "start", "topic": "ASL grammar basics"}

// User interruption (text)
{"type": "interrupt", "message": "Can you explain that again?"}

// Webcam frame (base64)
{"type": "frame", "image": "data:image/jpeg;base64,..."}

// Control commands
{"type": "pause"}
{"type": "resume"}
{"type": "skip"}  // Skip to next subtopic
{"type": "end"}
```

### Server → Client
```json
// Agent speaking
{
  "type": "speech",
  "agent": "rick",
  "text": "Listen Morty, *burp* Deaf culture isn't just about not hearing...",
  "audio_url": "/audio/12345.mp3"
}

// Morty responding
{
  "type": "speech", 
  "agent": "morty",
  "text": "Oh geez Rick, so you're saying it's like a whole identity thing?",
  "audio_url": "/audio/12346.mp3"
}

// Sign language detected
{
  "type": "sign_detected",
  "sign": "HELLO",
  "confidence": 0.92
}

// State change
{"type": "state", "state": "discussing", "topic": "ASL grammar"}

// Request topic
{"type": "request_topic"}

// Podcast ended
{"type": "ended", "summary": "..."}
```

## Sign Language Recognition Pipeline

```
Webcam Frame (30fps)
       │
       ▼
┌─────────────────┐
│  MediaPipe      │  → Extract hand landmarks (21 points × 2 hands)
│  Hands/Pose     │  → Extract body pose (for signs using arms)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  → Normalize coordinates
│                 │  → Create feature vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Sign Classifier│  → LSTM/Transformer for temporal signs
│  (Custom Model) │  → CNN for static signs (fingerspelling)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gesture Buffer │  → Accumulate frames for sequence signs
│  & Smoothing    │  → Confidence thresholding
└────────┬────────┘
         │
         ▼
  Recognized Sign
  (sent to Coordinator)
```

## File Structure (New)

```
/podcast/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── base.py           # Base Agent class
│   ├── summer.py         # Summer - coordinator/producer
│   ├── morty.py          # Morty - the curious host 😰
│   └── rick.py           # Rick - the genius expert 🥒
├── services/
│   ├── __init__.py
│   ├── tts_service.py    # Multi-voice TTS
│   ├── conversation.py   # Conversation memory
│   └── state_machine.py  # Podcast state management
├── vision/
│   ├── __init__.py
│   ├── webcam.py         # Frame capture & processing
│   ├── hand_detector.py  # MediaPipe integration
│   └── sign_classifier.py # Sign language recognition model
└── websocket/
    ├── __init__.py
    └── handler.py        # WebSocket connection management

/frontend/src/
├── components/
│   ├── PodcastView.jsx   # Main podcast UI
│   ├── AgentAvatar.jsx   # Animated agent avatars
│   ├── WebcamCapture.jsx # Webcam component
│   ├── TopicInput.jsx    # Topic suggestion UI
│   └── TranscriptPanel.jsx # Live transcript
└── hooks/
    └── useWebSocket.js   # WebSocket hook
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | Custom async Python (or LangGraph) |
| WebSocket | FastAPI WebSocket |
| TTS | ElevenLabs (different voices per agent) |
| Hand Detection | MediaPipe Hands |
| Sign Recognition | Custom PyTorch model (LSTM/Transformer) |
| Frontend Webcam | WebRTC / Canvas API |
| State Management | Python asyncio + state machine |

## Example Conversation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎙️ SIGNTUTOR PODCAST - Episode: "What is Deaf Culture?"                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  🎯 SUMMER: "Alright, the user wants to learn about Deaf culture.           │
│              Rick, Morty—try not to go on any tangents this time."          │
│                                                                              │
│  😰 MORTY: "Oh, oh geez Rick, so like... Deaf culture? I-I thought          │
│             being deaf was just, you know, not being able to hear?"         │
│                                                                              │
│  🥒 RICK: "Morty, Morty, Morty. *burp* That's the most reductive thing      │
│           I've heard since the Council of Ricks tried to define             │
│           consciousness. Look—'Deaf' with a capital D isn't about           │
│           what you CAN'T do. It's a whole linguistic and cultural           │
│           identity, Morty!"                                                  │
│                                                                              │
│  😰 MORTY: "W-wait, so there's like a difference between 'deaf' and         │
│             'Deaf'? That's... that's actually pretty interesting."          │
│                                                                              │
│  🥒 RICK: "NOW you're getting it! Lowercase 'deaf' is the audiological      │
│           condition. Capital D 'Deaf' means you're part of the community,   │
│           you use sign language, you share values and history. It's like—   │
│           *burp*—it's like how 'american' vs 'American' matters, Morty."    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  👋 USER INTERRUPTION DETECTED (via webcam sign: "QUESTION")         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  🎯 SUMMER: "Hold up—the user has a question."                              │
│                                                                              │
│  👤 USER (via sign): [WHAT] [ABOUT] [CODA]                                  │
│                                                                              │
│  🎯 SUMMER: "They're asking about CODAs."                                   │
│                                                                              │
│  😰 MORTY: "Coda? Like... like in music? The ending part?"                  │
│                                                                              │
│  🥒 RICK: "Different CODA, Morty. C-O-D-A. Child of Deaf Adult.             │
│           These are hearing kids raised by Deaf parents. They grow up       │
│           bilingual, bicultural. Some of the best interpreters come         │
│           from CODA families. It's actually *burp* fascinating from a       │
│           linguistic acquisition standpoint..."                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 2.1: Core Agent Infrastructure
- [ ] Create base Agent class with LLM integration
- [ ] Implement Rick 🥒 (Expert) and Morty 😰 (Host) agents
- [ ] Set up WebSocket server
- [ ] Basic conversation loop with characteristic dialogue

### Phase 2.2: Coordinator & State Machine
- [ ] Implement Summer (Coordinator/Producer) agent
- [ ] State machine for conversation flow
- [ ] User interruption handling
- [ ] Topic management

### Phase 2.3: Visual Recognition
- [ ] Webcam frame capture in frontend
- [ ] MediaPipe hand detection backend
- [ ] Sign language classifier model
- [ ] Integration with Coordinator

### Phase 2.4: Polish & UI
- [ ] Podcast-style React UI
- [ ] Agent avatars with speaking animations
- [ ] Live transcript
- [ ] Audio playback queue

