# SignTutor Demo Video Script
## 2-Minute Technical Walkthrough

---

## **SEGMENT 1: Project Overview (0:00-0:20)**

"Welcome to SignTutor - an agentic AI-powered podcast platform for learning sign language.

This project combines **14,745 lines of code** across Python backend, React frontend, and computer vision modules to create an interactive learning experience powered by multiple AI agents.

Let's dive into the technical architecture."

---

## **SEGMENT 2: Backend Architecture (0:20-0:50)**

**[Show: Backend code structure]**

"The backend is built with **FastAPI** - a modern Python web framework running on **uvicorn**.

**Core Statistics:**
- **2,531 lines** of podcast agent logic
- **15 Python libraries** including:
  - OpenAI for cloud LLM
  - Transformers & PEFT for local Qwen3-4B model
  - FastAPI for REST API and WebSocket support
  - Pydantic for type-safe data validation

**Multi-Agent System:**
The podcast is powered by **3 coordinated AI agents**:

1. **Rick Agent** - Expert with deep knowledge
2. **Morty Agent** - Curious learner who asks questions  
3. **Summer Coordinator** - Manages turns and handles interrupts

These agents use a **state machine** with 4 states:
- `topic_input` → `discussing` → `interrupted` → `ended`

**Agent Workflow:**
```
User starts → Summer chooses agent → Agent speaks → Summer evaluates → Next agent
                                                    ↓
                                            User interrupts (vision/text)
```

Each agent generates responses using either:
- **Cloud**: GPT-4o-mini via OpenAI API
- **Local**: Fine-tuned Qwen3-4B (4 billion parameters, LoRA adapters)

**Text-to-Speech Services:**
- **Google Cloud TTS**: WaveNet voices for podcast agents
- **ElevenLabs**: Premium character voices (Rick & Morty modes)
- **Browser TTS**: Fallback for offline use

All running through **2 WebSocket endpoints**:
- `/ws/podcast` - Agent communication
- `/ws/vision` - ASL recognition stream"

---

## **SEGMENT 3: Frontend Architecture (0:50-1:15)**

**[Show: React component tree]**

"The frontend is built with **React + Vite** for fast hot-module replacement.

**Code Statistics:**
- **3,687 lines** across JSX, JS, and CSS
- **16 npm packages** including:
  - React 18 with hooks
  - Axios for HTTP/WebSocket
  - React-Markdown with remark-gfm for formatted responses

**Key Components:**

1. **PodcastView** (726 lines)
   - Manages WebSocket connection
   - Speech queue for sequential TTS playback
   - Personality switching (Fun vs Professional mode)
   - Model selection (Cloud vs Local)

2. **VisionSidebar** (280 lines)
   - Real-time webcam capture at 10 FPS
   - WebSocket streaming to vision backend
   - Letter buffer with debouncing (0.5s, 70% confidence threshold)
   - Word suggestions from 200+ dictionary
   - One-click podcast interrupts

3. **App.jsx** (652 lines)
   - Chat interface with streaming text
   - Settings modal for TTS configuration
   - View switching between chat and podcast modes

**State Management:**
Uses React hooks (useState, useEffect, useCallback, useRef) for:
- WebSocket connection status
- Agent speaking state
- Letter recognition buffer
- TTS playback queue"

---

## **SEGMENT 4: Vision Pipeline (1:15-1:50)**

**[Show: Vision recognition flow diagram]**

"The **ASL Vision System** adds **6,418 lines** of computer vision code.

**Recognition Pipeline:**
```
Webcam (10 FPS) → YOLO Hand Detection → ResNet34 Classifier → Letter Buffer → Word Formation
```

**Models:**
1. **YOLO (6MB)**: Real-time hand detection with keypoint extraction
2. **ResNet34 (247MB)**: 24-class ASL letter classifier (A-Y, excluding J/Z)

**Training Data:**
- **55,707 total images**:
  - 27,456 from Sign MNIST (28×28 grayscale)
  - 28,251 from real-world collection (224×224 RGB via YOLO)

**Backend Services** (Python):
- `LetterRecognizer` - Processes base64 frames via WebSocket
- `WordBuilder` - Buffers letters with:
  - Debouncing to prevent repeated letters
  - Confidence filtering (>70%)
  - Dictionary-based word suggestions

**Real-time Flow:**
1. Frontend captures frame at 10 FPS
2. Converts to base64 JPEG (70% quality)
3. Sends via WebSocket to `/ws/vision`
4. Backend runs YOLO → crops hand → ResNet34 prediction
5. Returns letter + confidence + suggestions
6. Frontend updates buffer and displays word

**Integration with Podcast:**
When user signs a word like "HELLO", they can click "Send Question" which:
- Stops current agent speech
- Sends interrupt to Summer coordinator
- Resumes conversation with user's ASL question"

---

## **SEGMENT 5: Demo Highlights (1:50-2:00)**

**[Show: Live demo clips]**

"**Complete Workflow:**
1. Start podcast with topic (e.g., "Deaf Culture")
2. Agents discuss collaboratively with turn-taking
3. User signs ASL letters (real-time recognition visible)
4. Letters buffer into words with suggestions
5. User interrupts podcast with signed question
6. Agents respond naturally and continue discussion

**Technical Achievement:**
- 3 AI agents coordinating via state machine
- 2 deep learning models (LLM + Vision) working in real-time
- Seamless integration of 4 services (OpenAI, Google TTS, YOLO, ResNet34)
- Full-stack implementation across 14,745 lines of code

Thank you for watching!"

---

## **Key Technical Metrics Summary**

| Component | Lines of Code | Key Technologies |
|-----------|--------------|------------------|
| **Backend (Podcast)** | 2,531 | FastAPI, OpenAI, Transformers, Pydantic |
| **Frontend** | 3,687 | React 18, Vite, Axios, Markdown |
| **Vision System** | 6,418 | PyTorch, Ultralytics YOLO, OpenCV |
| **Total Project** | **14,745** | **31 libraries** (15 Python + 16 npm) |

| AI Models | Parameters | Purpose |
|-----------|-----------|---------|
| Qwen3-4B (Local) | 4 billion | Podcast agent responses |
| GPT-4o-mini (Cloud) | ~8 billion | Faster cloud alternative |
| ResNet34 (Vision) | 21.56 million | ASL letter classification |
| YOLO (Hand Detection) | ~6 million | Real-time hand localization |

| Services | Provider | Usage |
|----------|----------|--------|
| Text-to-Speech | Google Cloud TTS | Agent voices (WaveNet) |
| TTS Premium | ElevenLabs | Character voices (Rick/Morty) |
| LLM Cloud | OpenAI GPT-4o-mini | Fast agent responses |
| Vision Hosting | Local GPU | Real-time ASL recognition |

**Data Pipeline:**
- 28,251 hand images collected via YOLO
- 27,456 Sign MNIST images (Kaggle dataset)
- **55,707 total training images**
- Real-time inference at 10 FPS

---

## **Recording Notes**

**Screen Recording Sections:**
1. Code walkthrough (VSCode):
   - Show `podcast/podcast_manager.py` - state machine
   - Show `podcast/agents.py` - agent personalities
   - Show `app.py` - WebSocket endpoints
   - Show `frontend/src/PodcastView.jsx` - React component
   - Show `vision/services/letter_recognizer.py` - vision pipeline

2. Live Demo:
   - Start podcast
   - Show agent turn-taking
   - Open vision sidebar
   - Sign letters (show real-time recognition)
   - Interrupt with ASL question
   - Show response flow

**Timing Breakdown:**
- Intro: 20s
- Backend: 30s
- Frontend: 25s
- Vision: 35s
- Demo: 10s
**Total: 2:00**
