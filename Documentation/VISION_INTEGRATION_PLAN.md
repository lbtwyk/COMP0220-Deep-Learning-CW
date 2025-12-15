# Vision Integration Implementation Plan

## 📋 Overview

Integrate real-time ASL fingerspelling recognition into the SignTutor podcast system, allowing users to sign letters that are concatenated into words and sent as interruptions to the podcast agents.

**Status**: ⏳ Waiting for ResNet34 model training to complete  
**Training Progress**: Running for 20+ minutes, model checkpoint saved (247MB)  
**Implementation Timeline**: 1-2 days after model is ready

---

## 🎯 Core Features

### 1. Real-time Letter Recognition
- **Frame Rate**: 10-15 FPS (optimized for performance)
- **Model**: ResNet34 trained on 28K+ hand-collected ASL images
- **Hand Detection**: YOLO v8 pose model
- **Confidence Threshold**: >70% for letter acceptance

### 2. Word Formation System
- **Letter Buffering**: Accumulate recognized letters
- **Debouncing**: Require 0.5s stability before adding letter
- **Word Suggestions**: Dictionary lookup for auto-complete
- **Manual Control**: User confirms before sending

### 3. UI Integration
- **Sidebar Panel**: Non-intrusive webcam preview + letter display
- **Send Mechanism**: Manual button (primary) + optional auto-send
- **Visual Feedback**: Confidence bars, letter buffer display

---

## 🏗️ Architecture

### System Flow

```
┌─────────────┐
│  Webcam     │ 10-15 FPS
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Backend: Vision Pipeline                       │
│  ┌─────────────────────────────────────────┐   │
│  │  1. YOLO Hand Detection                 │   │
│  │     - Detect hand bounding box          │   │
│  │     - Extract hand ROI (224x224)        │   │
│  └─────────────┬───────────────────────────┘   │
│                ▼                                │
│  ┌─────────────────────────────────────────┐   │
│  │  2. ResNet34 Letter Classifier          │   │
│  │     - Predict letter (a-y, 24 classes)  │   │
│  │     - Return confidence score           │   │
│  └─────────────┬───────────────────────────┘   │
│                ▼                                │
│  ┌─────────────────────────────────────────┐   │
│  │  3. Letter Buffer Service               │   │
│  │     - Debounce repeated letters         │   │
│  │     - Accumulate letter sequence        │   │
│  │     - Generate word suggestions         │   │
│  └─────────────┬───────────────────────────┘   │
└────────────────┼───────────────────────────────┘
                 │ WebSocket
                 ▼
┌─────────────────────────────────────────────────┐
│  Frontend: Sidebar UI                           │
│  ┌─────────────────────────────────────────┐   │
│  │  📹 Webcam Preview (small)               │   │
│  │  Current: H (85% confidence)            │   │
│  │  Buffer: [H][E][L][L][O]                │   │
│  │  Suggestions: HELLO ✓, HELP             │   │
│  │  [Clear] [Send Question]                │   │
│  └─────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────┘
                  │ Interrupt Message
                  ▼
┌─────────────────────────────────────────────────┐
│  Podcast Agent: Summer (Coordinator)            │
│  - Receives: "What does HELLO mean in ASL?"    │
│  - Triggers: USER_INTERRUPT state              │
│  - Responds via Rick/Morty                     │
└─────────────────────────────────────────────────┘
```

---

## 📦 Components to Build

### Backend Services

#### 1. `vision/services/letter_recognizer.py`
```python
class LetterRecognizer:
    """Real-time ASL letter recognition service"""
    
    def __init__(self):
        self.yolo_model = YOLO("vision/yolo/ckpt/best.pt")
        self.asl_model = load_resnet34("vision/ASL/models/best_asl_model_resnet34.pth")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def recognize_frame(self, frame_bytes):
        """
        Process single webcam frame
        Returns: (letter: str, confidence: float, hand_bbox: tuple)
        """
        # 1. Decode frame
        # 2. YOLO hand detection
        # 3. Extract hand ROI
        # 4. ResNet34 classification
        # 5. Return results
```

**Key Features**:
- Frame decoding (base64 → numpy/PIL)
- Hand detection with confidence filtering (>0.4)
- Hand ROI extraction with margin
- Letter prediction with confidence threshold
- Error handling for no-hand scenarios

---

#### 2. `vision/services/word_builder.py`
```python
class WordBuilder:
    """Buffer letters and suggest words"""
    
    def __init__(self):
        self.buffer = []  # ['H', 'E', 'L', 'L', 'O']
        self.last_letter = None
        self.last_letter_time = None
        self.debounce_threshold = 0.5  # seconds
        self.word_dict = self.load_dictionary()
    
    def add_letter(self, letter, confidence, timestamp):
        """Add letter with debouncing"""
        # Only add if:
        # - Confidence > 0.7
        # - Different from last letter OR 0.5s elapsed
        
    def get_suggestions(self):
        """Return matching words from dictionary"""
        # Check buffer against word list
        # Return top 3 matches
        
    def clear_buffer(self):
        """Reset buffer"""
        
    def get_current_word(self):
        """Return buffer as string"""
```

**Key Features**:
- Debouncing to prevent "AAAA" from holding "A"
- Dictionary lookup (use NLTK words or custom ASL vocab)
- Confidence-based filtering
- Buffer management (clear, delete last, etc.)

---

#### 3. `app.py` - WebSocket Endpoint
```python
@app.websocket("/ws/vision")
async def vision_websocket(websocket: WebSocket):
    """Handle vision recognition WebSocket"""
    await websocket.accept()
    
    recognizer = LetterRecognizer()
    word_builder = WordBuilder()
    
    try:
        while True:
            # Receive frame from frontend
            data = await websocket.receive_json()
            
            if data['type'] == 'frame':
                # Process frame
                letter, conf = recognizer.recognize_frame(data['image'])
                
                if letter:
                    word_builder.add_letter(letter, conf, time.time())
                    
                    # Send update to frontend
                    await websocket.send_json({
                        'type': 'letter_update',
                        'letter': letter,
                        'confidence': conf,
                        'buffer': word_builder.buffer,
                        'suggestions': word_builder.get_suggestions()
                    })
            
            elif data['type'] == 'clear_buffer':
                word_builder.clear_buffer()
            
            elif data['type'] == 'send_interrupt':
                # Send to podcast WebSocket
                # (integrate with existing interrupt handler)
                pass
                
    except WebSocketDisconnect:
        pass
```

---

### Frontend Components

#### 4. `frontend/src/components/VisionSidebar.jsx`
```jsx
import React, { useState, useEffect, useRef } from 'react';
import { useWebcam } from '../hooks/useWebcam';
import { useVisionWebSocket } from '../hooks/useVisionWebSocket';

export const VisionSidebar = ({ onInterrupt }) => {
  const [buffer, setBuffer] = useState([]);
  const [currentLetter, setCurrentLetter] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [suggestions, setSuggestions] = useState([]);
  
  const videoRef = useRef();
  const { stream } = useWebcam();
  const { sendFrame, clearBuffer } = useVisionWebSocket({
    onLetterUpdate: (data) => {
      setCurrentLetter(data.letter);
      setConfidence(data.confidence);
      setBuffer(data.buffer);
      setSuggestions(data.suggestions);
    }
  });
  
  // Capture and send frames at 10 FPS
  useEffect(() => {
    const interval = setInterval(() => {
      if (videoRef.current) {
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, 224, 224);
        const base64 = canvas.toDataURL('image/jpeg').split(',')[1];
        sendFrame(base64);
      }
    }, 100); // 10 FPS
    
    return () => clearInterval(interval);
  }, []);
  
  const handleSend = () => {
    const message = `What does ${buffer.join('')} mean in ASL?`;
    onInterrupt(message, 'vision');
    clearBuffer();
  };
  
  return (
    <div className="vision-sidebar">
      <h3>📹 Sign Language Input</h3>
      
      <video ref={videoRef} autoPlay playsInline />
      
      <div className="current-letter">
        <span>{currentLetter || '—'}</span>
        <progress value={confidence} max={1}>{(confidence * 100).toFixed()}%</progress>
      </div>
      
      <div className="letter-buffer">
        {buffer.map((letter, i) => (
          <span key={i} className="letter">{letter}</span>
        ))}
      </div>
      
      {suggestions.length > 0 && (
        <div className="suggestions">
          <strong>💡 Suggestions:</strong>
          {suggestions.map(word => (
            <button key={word} onClick={() => handleSelectSuggestion(word)}>
              {word}
            </button>
          ))}
        </div>
      )}
      
      <div className="controls">
        <button onClick={clearBuffer}>Clear</button>
        <button onClick={handleSend} disabled={buffer.length === 0}>
          Send Question
        </button>
      </div>
    </div>
  );
};
```

---

#### 5. `frontend/src/hooks/useVisionWebSocket.js`
```javascript
import { useState, useEffect, useCallback } from 'react';

export const useVisionWebSocket = ({ onLetterUpdate }) => {
  const [ws, setWs] = useState(null);
  
  useEffect(() => {
    const socket = new WebSocket('ws://localhost:8000/ws/vision');
    
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'letter_update') {
        onLetterUpdate(data);
      }
    };
    
    setWs(socket);
    return () => socket.close();
  }, []);
  
  const sendFrame = useCallback((base64Image) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'frame',
        image: base64Image
      }));
    }
  }, [ws]);
  
  const clearBuffer = useCallback(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'clear_buffer' }));
    }
  }, [ws]);
  
  return { sendFrame, clearBuffer };
};
```

---

## 🔄 Integration with Existing Podcast System

### Minimal Changes Required

**1. Summer Agent (`podcast/agents/summer.py`)**
```python
# Already has interrupt handling, just ensure vision source is supported
async def handle_interrupt(self, message, source='text'):
    if source == 'vision':
        # Maybe add special handling or logging
        print(f"Vision interrupt: {message}")
    
    # Existing interrupt logic works as-is
    self.state_machine.transition_to(PodcastState.USER_INTERRUPT)
    # ... rest of existing code
```

**2. WebSocket Handler (`app.py`)**
```python
# Add vision interrupt routing to existing podcast WebSocket
@app.websocket("/ws/podcast")
async def podcast_websocket(websocket: WebSocket):
    # ... existing code ...
    
    if data['type'] == 'vision_interrupt':
        # Route to Summer agent
        await podcast_manager.handle_interrupt(
            message=data['message'],
            source='vision'
        )
```

---

## 🎨 UI/UX Design

### Sidebar Placement
```
┌─────────────────────────────────────────────────────┐
│  SignTutor Podcast                    [Vision] [•••]│
├─────────────────┬───────────────────────────────────┤
│                 │                                   │
│  Podcast View   │  📹 Sign Language Input          │
│                 │  ┌─────────────────────────┐     │
│  [Dave Avatar]  │  │   Webcam Preview        │     │
│  Speaking...    │  │   (224x224)             │     │
│                 │  └─────────────────────────┘     │
│  [Taylor]       │                                   │
│  Listening      │  Current: H (85%)                │
│                 │  ████████░░░░░░░░                │
│                 │                                   │
│  [Transcript]   │  Buffer: H E L L O               │
│  Dave: ASL is   │                                   │
│  more than...   │  💡 HELLO ✓, HELP                 │
│                 │                                   │
│                 │  [Clear] [Send Question]          │
│                 │                                   │
└─────────────────┴───────────────────────────────────┘
```

### Visual States

**Idle (No Hand Detected)**
```
Current: — (0%)
░░░░░░░░░░░░░░░░
```

**Low Confidence (<70%)**
```
Current: H (45%) [LOW]
████░░░░░░░░░░░░ (orange)
```

**High Confidence (>70%)**
```
Current: H (85%)
████████░░░░░░░░ (green)
```

---

## ⚡ Performance Optimizations

### 1. Smart Frame Sampling
```python
class SmartSampler:
    def should_process_frame(self, hand_detected, buffer_active):
        if not hand_detected:
            return every_nth_frame(n=5)  # 3 FPS when idle
        elif buffer_active:
            return every_nth_frame(n=2)  # 15 FPS when signing
        else:
            return every_nth_frame(n=3)  # 10 FPS default
```

### 2. GPU Memory Management
```python
# Only load one model at a time if memory constrained
if not podcast_using_gpu:
    load_vision_models_on_gpu()
else:
    load_vision_models_on_cpu()
```

### 3. Frame Compression
```python
# Reduce JPEG quality for transmission
canvas.toDataURL('image/jpeg', 0.7)  # 70% quality
```

---

## 🧪 Testing Strategy

### Phase 1: Model Validation
- [ ] Test ResNet34 on held-out validation set
- [ ] Measure per-letter accuracy
- [ ] Identify confusing letter pairs (e.g., M vs N)
- [ ] Test with different lighting conditions

### Phase 2: Integration Testing
- [ ] Test YOLO + ResNet34 pipeline end-to-end
- [ ] Verify debouncing works correctly
- [ ] Test word buffer with various inputs
- [ ] Validate WebSocket communication

### Phase 3: User Experience Testing
- [ ] Test with different users (hand sizes/skin tones)
- [ ] Measure latency (frame → letter display)
- [ ] Test auto-complete accuracy
- [ ] Verify interrupt → podcast flow

---

## 📊 Success Metrics

### Technical Metrics
- **Letter Recognition Accuracy**: >85% on validation set
- **Frame Processing Latency**: <100ms per frame
- **End-to-End Latency**: <300ms (sign → display)
- **False Positive Rate**: <5% (wrong letters added to buffer)

### UX Metrics
- **User Satisfaction**: Can successfully ask questions
- **Completion Rate**: >80% of signed words recognized correctly
- **Error Recovery**: Easy to clear/fix mistakes

---

## 🚀 Implementation Phases

### Phase 1: Backend Foundation (2-4 hours)
- [x] Train ResNet34 model
- [ ] Create `LetterRecognizer` service
- [ ] Create `WordBuilder` service
- [ ] Add `/ws/vision` endpoint to `app.py`
- [ ] Test with curl/Postman

### Phase 2: Frontend Integration (3-5 hours)
- [ ] Create `VisionSidebar` component
- [ ] Implement `useVisionWebSocket` hook
- [ ] Add webcam capture logic
- [ ] Style sidebar UI
- [ ] Test in isolation

### Phase 3: Podcast Integration (2-3 hours)
- [ ] Connect vision sidebar to main podcast app
- [ ] Route vision interrupts to Summer agent
- [ ] Test full workflow: sign → recognize → interrupt → response
- [ ] Handle edge cases (disconnect, errors, etc.)

### Phase 4: Polish & Optimization (2-3 hours)
- [ ] Optimize frame rate based on GPU usage
- [ ] Add visual feedback animations
- [ ] Implement word suggestions
- [ ] Add gesture controls (optional)
- [ ] Write user documentation

**Total Estimated Time**: 9-15 hours

---

## 🐛 Known Limitations & Future Enhancements

### Current Limitations
- **Static letters only**: J and Z (motion-based) not supported
- **Single hand**: Doesn't handle two-handed signs
- **English vocabulary**: Limited to common English words
- **Lighting sensitive**: Performance degrades in poor lighting

### Future Enhancements
- **Motion recognition**: Add LSTM for J, Z, and word signs
- **Two-handed gestures**: Detect compound signs
- **ASL vocabulary**: Add ASL-specific words and phrases
- **Multi-user**: Support multiple simultaneous users
- **Offline mode**: Cache dictionary for offline use

---

## 📚 Dependencies

### Python Packages (Backend)
```txt
torch>=2.1.0
torchvision>=0.16.0
opencv-python>=4.8.0
ultralytics>=8.0.0
pillow>=10.0.0
numpy>=1.24.0
nltk>=3.8  # For word dictionary
```

### JavaScript Packages (Frontend)
```json
{
  "react-webcam": "^7.1.1",  // Webcam access
  "canvas": "^2.11.2"         // Frame encoding (if needed)
}
```

---

## 📝 Documentation to Update

After implementation:
- [ ] Update `PODCAST_BACKEND_GUIDE.md` with vision endpoints
- [ ] Update `VISION_INTEGRATION_GUIDE.md` with usage instructions
- [ ] Add vision demo to `README.md`
- [ ] Create video tutorial (2 minutes)
- [ ] Document troubleshooting guide

---

## 🎬 Demo Requirements

### 8-Minute Podcast Recording
**Topics to cover**:
1. Introduction to ASL fingerspelling (2 min)
2. Difference between Deaf/deaf (2 min)
3. ASL grammar basics (2 min)
4. User interrupts with vision input (1 min)
5. Wrap-up and summary (1 min)

**Demonstrate**:
- Natural conversation flow between Rick/Morty
- Summer coordinating topic changes
- Vision sidebar recognizing letters
- Successful interrupt from signed word

### 2-Minute Explanation Video
**Structure**:
1. **Frontend Walkthrough** (45s)
   - Show podcast interface
   - Explain agent avatars (Rick/Morty/Summer)
   - Demonstrate vision sidebar

2. **Agentic Workflow** (45s)
   - Explain multi-agent conversation
   - Show state machine transitions
   - Demonstrate interrupt handling

3. **Technical Highlights** (30s)
   - ASL recognition pipeline
   - Real-time letter detection
   - Integration with podcast agents

---

## ✅ Checklist Before Implementation

- [x] ResNet34 model trained and saved
- [ ] Model validation accuracy >85%
- [ ] Real-time inference tested (manual)
- [ ] WebSocket infrastructure reviewed
- [ ] Frontend React components planned
- [ ] Integration points identified
- [ ] Testing strategy defined

**Status**: Ready to implement after model training completes! 🚀

---

## 📞 Contact & Support

For questions during implementation:
- Check `VISION_INTEGRATION_GUIDE.md` for technical details
- Review `PODCAST_ARCHITECTURE.md` for system design
- Test each component independently before integration
- Use `nvidia-smi` to monitor GPU usage during development

---

**Last Updated**: 2025-12-15  
**Next Review**: After model training completion
