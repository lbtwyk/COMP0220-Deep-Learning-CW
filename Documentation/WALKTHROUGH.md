# SignTutor Project Walkthrough
## Complete System Documentation for Demo Recording

---

## **Quick Start Guide**

### Starting the Application

**Terminal 1 - Backend:**
```bash
cd /home/hz/COMP0220\ DL/COMP0220-Deep-Learning-CW
conda activate comp0220
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd /home/hz/COMP0220\ DL/COMP0220-Deep-Learning-CW/frontend
npm run dev
```

**Access:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## **Demo Flow for 8-Minute Podcast Recording**

### Recommended Topics:
1. **"What is Deaf Culture?"** - Broad discussion covering community, identity, language
2. **"How does ASL grammar differ from English?"** - Technical linguistic topic
3. **"What does CODA mean?"** - Specific term explanation with cultural context
4. **"History of sign language recognition"** - Historical + activist perspective

### Recording Steps:
1. Open browser to http://localhost:5173
2. Click "Podcast Mode" banner
3. Select personality (Professional recommended for clarity)
4. Click "Connect to Podcast"
5. Enter topic in text box
6. **Record for 8 minutes** showing:
   - Natural conversation flow between agents
   - Turn-taking behavior
   - Agent interruption (text input)
   - TTS playback
   - Model badges (Cloud/Local)

### Demo Features to Highlight:
- ✅ Agent turn-taking and coordination
- ✅ Natural conversation flow
- ✅ Text interruptions
- ✅ TTS voice quality
- ✅ Personality switching (Fun vs Professional)
- ✅ Model selection (Cloud vs Local)

---

## **Demo Flow for 2-Minute Explanation Video**

Follow the script in `DEMO_VIDEO_SCRIPT.md` with these visual aids:

### Section 1: Overview (0:00-0:20)
**Show:** Project folder structure in VSCode
**Mention:** 14,745 lines of code across 3 components

### Section 2: Backend (0:20-0:50)
**Show these files in order:**
1. `app.py` - FastAPI endpoints, WebSocket handlers
2. `podcast/podcast_manager.py` - State machine (lines 80-150)
3. `podcast/agents.py` - Agent classes (RickAgent, MortyAgent, SummerAgent)
4. `requirements.txt` - Libraries list

**Key code to highlight:**
```python
# State Machine in podcast_manager.py
self.state = "idle" | "topic_input" | "discussing" | "interrupted"

# Agent coordination
current_agent = self.choose_next_agent()
response = await agent.generate_response(context)
```

### Section 3: Frontend (0:50-1:15)
**Show these files:**
1. `frontend/src/App.jsx` - Main chat interface
2. `frontend/src/components/PodcastView.jsx` - Podcast component
3. `frontend/src/components/VisionSidebar.jsx` - Vision integration
4. `frontend/package.json` - Dependencies

**Key React patterns:**
```jsx
// WebSocket connection
const [ws, setWs] = useState(null)
const [messages, setMessages] = useState([])

// TTS queue management
const queueSpeech = (text, agent) => {
  speechQueue.push({ text, agent })
  processSpeechQueue()
}
```

### Section 4: Vision Pipeline (1:15-1:50)
**Show these files:**
1. `vision/services/letter_recognizer.py` - YOLO + ResNet34 pipeline
2. `vision/services/word_builder.py` - Letter buffering logic
3. `vision/ASL/models/` - Model files and sizes

**Diagram to show:**
```
Webcam (10fps) → base64 encode → WebSocket 
                                      ↓
                          /ws/vision endpoint
                                      ↓
                  YOLO (hand detection) → crop
                                      ↓
                  ResNet34 (classify) → letter
                                      ↓
              WordBuilder (buffer + suggest)
                                      ↓
              Frontend (display + interrupt)
```

### Section 5: Live Demo (1:50-2:00)
**Quick demo showing:**
1. Start podcast
2. Open vision sidebar
3. Sign a few letters (real-time recognition)
4. Form word "HELLO"
5. Click "Send Question"
6. Show interrupt working

---

## **Technical Architecture Diagram**

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   App.jsx    │  │ PodcastView  │  │ VisionSidebar│  │
│  │  (Chat UI)   │  │  (3 Agents)  │  │  (ASL Input) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                  │                  │          │
└─────────┼──────────────────┼──────────────────┼──────────┘
          │                  │                  │
          │ HTTP/WS          │ WS               │ WS
          ↓                  ↓                  ↓
┌─────────────────────────────────────────────────────────┐
│              BACKEND (FastAPI + Python)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   /chat      │  │ /ws/podcast  │  │  /ws/vision  │  │
│  │   /tts       │  │ PodcastMgr   │  │ Recognizer   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                  │                  │          │
└─────────┼──────────────────┼──────────────────┼──────────┘
          │                  │                  │
          ↓                  ↓                  ↓
    ┌─────────┐      ┌──────────────┐   ┌─────────────┐
    │ OpenAI  │      │ Rick Agent   │   │ YOLO        │
    │   API   │      │ Morty Agent  │   │ (6MB)       │
    │         │      │ Summer Agent │   │ ResNet34    │
    │ Google  │      │              │   │ (247MB)     │
    │   TTS   │      │ Qwen3-4B or  │   │             │
    │         │      │ GPT-4o-mini  │   │ 55K images  │
    │ElevenLabs      └──────────────┘   └─────────────┘
    └─────────┘
     (Services)        (AI Agents)       (Vision Models)
```

---

## **Project Statistics Summary**

### Code Metrics
| Component | Lines | Files | Purpose |
|-----------|-------|-------|---------|
| Backend Podcast | 2,531 | 8 | Agent logic, state machine, WebSocket |
| Frontend | 3,687 | 7 | React UI, components, styling |
| Vision System | 6,418 | 12 | YOLO, ResNet34, training, inference |
| **Total** | **14,745** | **27+** | Full-stack ASL learning platform |

### Dependencies
| Category | Count | Key Libraries |
|----------|-------|---------------|
| Python Backend | 15 | FastAPI, OpenAI, Transformers, PyTorch |
| Frontend npm | 16 | React, Vite, Axios, React-Markdown |
| **Total** | **31** | Modern tech stack |

### AI Models
| Model | Size | Parameters | Usage |
|-------|------|------------|-------|
| Qwen3-4B | ~8GB | 4B | Local podcast responses |
| GPT-4o-mini | N/A | ~8B | Cloud podcast responses |
| ResNet34 | 247MB | 21.56M | ASL letter classification |
| YOLO v8 | 6MB | ~6M | Hand detection/keypoints |

### Training Data
| Dataset | Images | Format | Source |
|---------|--------|--------|--------|
| Sign MNIST | 27,456 | 28×28 gray | Kaggle public dataset |
| Collected Data | 28,251 | 224×224 RGB | YOLO auto-collection |
| **Combined** | **55,707** | Normalized | Used for ResNet34 training |

### External Services
| Service | Provider | Purpose | Cost |
|---------|----------|---------|------|
| LLM Cloud | OpenAI | GPT-4o-mini responses | Pay-per-token |
| TTS Cloud | Google Cloud | WaveNet voices | 4M chars/month free |
| TTS Premium | ElevenLabs | Character voices | 10K chars/month free |

---

## **Recording Checklist**

### Before Recording:
- [ ] Both servers running (backend + frontend)
- [ ] New combined model loaded (if training finished)
- [ ] Browser camera permissions granted
- [ ] Screen recording software ready (OBS/QuickTime)
- [ ] Audio input working (for narration)

### 8-Minute Podcast Recording:
- [ ] Start with clear topic
- [ ] Let agents discuss for 3-4 turns minimum
- [ ] Show at least 1 text interrupt
- [ ] Demonstrate personality switching
- [ ] Show model badge (Cloud/Local indicator)
- [ ] Display full conversation flow

### 2-Minute Explanation Recording:
- [ ] Follow script timing (5 segments)
- [ ] Show relevant code files for each section
- [ ] Include live demo at end
- [ ] Display architecture diagram
- [ ] Mention all key statistics
- [ ] End with working demo

### Optional Enhancements:
- [ ] Show training loss curves (if available)
- [ ] Display inference speed metrics
- [ ] Demonstrate vision accuracy with different hands
- [ ] Compare Cloud vs Local model responses
- [ ] Show concurrent user scenario (if time)

---

## **Troubleshooting Guide**

### Vision Not Working:
1. Check webcam not in use: `lsof /dev/video0`
2. Verify backend loaded models: Check uvicorn logs
3. Refresh browser (Ctrl+Shift+R)
4. Check browser console for errors

### Podcast Not Responding:
1. Check WebSocket connection status (should show "Connected")
2. Verify backend logs for errors
3. Ensure API keys set (if using cloud models)
4. Restart backend server

### TTS Not Playing:
1. Check TTS provider in settings
2. For Google TTS: Verify GOOGLE_APPLICATION_CREDENTIALS
3. For ElevenLabs: Check API key
4. Try browser TTS as fallback

### Model Switching:
1. Local model requires ~8GB VRAM
2. Cloud model needs OPENAI_API_KEY
3. Check model badge in UI to verify active model

---

## **Post-Recording Tasks**

1. **Stop All Services:**
   ```bash
   # Stop frontend (Ctrl+C in terminal)
   # Stop backend (Ctrl+C in terminal)
   # Stop training (if still running)
   pkill -f train_combined
   ```

2. **Export Videos:**
   - 8-minute podcast: `podcast_demo.mp4`
   - 2-minute explanation: `technical_walkthrough.mp4`

3. **Optional: Export Training Logs:**
   ```bash
   # Copy training logs
   cp vision/ASL/training/train_combined.log ./training_results.log
   
   # Check final model accuracy
   # (if training completed)
   ```

---

## **Project Highlights for Presentation**

**Innovation Points:**
1. **Multi-Agent Coordination** - 3 AI personalities working together
2. **Real-Time Vision Integration** - ASL recognition at 10 FPS
3. **Hybrid Model Support** - Seamless Cloud/Local switching
4. **Full-Stack Implementation** - 14.7K lines across 3 domains
5. **Production-Ready Architecture** - WebSockets, state machines, error handling

**Technical Challenges Solved:**
1. Sequential TTS playback queue
2. Agent turn-taking coordination
3. Real-time frame processing pipeline
4. Combined dataset training (MNIST + custom)
5. WebSocket state synchronization

**Future Enhancements:**
1. Support for J and Z (motion signs)
2. Multi-user podcast rooms
3. Recording/transcript export
4. Additional sign languages (BSL, etc.)
5. Mobile app version

---

## **Contact & Resources**

- **Project GitHub**: (Add if public)
- **Course**: COMP0220 Deep Learning
- **Models Used**: Qwen3-4B, GPT-4o-mini, ResNet34, YOLOv8
- **Datasets**: Sign MNIST (Kaggle), Custom collection (28K images)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-15  
**Status**: ✅ Ready for Demo Recording
