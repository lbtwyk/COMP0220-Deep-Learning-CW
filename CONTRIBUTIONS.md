# Github link
https://github.com/lbtwyk/COMP0220-Deep-Learning-CW.git

# Submission Video


# Team Contributions & Limitations

This document outlines team contributions and known limitations of the COMP0220 Deep Learning coursework.

---

## Team Contributions

### Hanshang Zhu (Hanshang)

**Focus: Frontend, Backend, and System Integration**

- Built complete React web application with Vite
- Designed podcast interface (`PodcastView.jsx`) and vision sidebar (`VisionSidebar.jsx`)
- Implemented real-time WebSocket communication and TTS integration
- Created FastAPI server (`app.py`) with REST and WebSocket endpoints
- Developed word builder service for letter buffering
- Integrated all components and created startup script (`start_app.sh`)

---

### Guanming Wang

**Focus: Computer Vision System**

- Trained YOLOv11 for hand detection with custom dataset
- Designed and trained ResNet18 ASL classifier (100% validation accuracy)
- Collected ASL letter dataset using YOLO-based pipeline
- Created training scripts and real-time inference pipeline
- Built letter recognition service (`letter_recognizer.py`)

---

### Yukun Wang

**Focus: Language Model Development**

- Selected and fine-tuned Qwen3-4B with LoRA
- Created fine-tuning script (`finetune_qwen3.py`)
- Curated educational dialogue dataset (~500 samples)
- Implemented local model inference service (`local_model.py`)
- Configured quantization for efficient deployment

---

## Shared Contributions

- All members contributed to documentation and architecture decisions
- Cross-component integration testing and demo preparation

---

## Known Limitations

### Vision System

| Limitation | Impact | Potential Solution |
|------------|--------|-------------------|
| **Limited dataset** - trained on single user | Low accuracy (20-40%) on different users | Collect multi-user data, transfer learning |
| **Static letters only** - no J, Z | Cannot recognize motion-based letters | Add temporal modeling (LSTM/Transformer) |
| **Camera dependency** | Performance varies with hardware | Adaptive preprocessing, calibration |

### Language Model

| Limitation | Impact | Potential Solution |
|------------|--------|-------------------|
| **4B model size** - too small for generalization | Repetitive/inconsistent responses | Use larger model (7B+), add RAG |
| **~500 training samples** | Limited topic coverage | Expand dataset, synthetic data |
| **2048 token context** | Loses context in long conversations | Summarization, longer context models |

### Frontend-Backend

| Limitation | Impact | Potential Solution |
|------------|--------|-------------------|
| **No RAG** - no web search | Cannot access up-to-date information | Integrate search APIs |
| **Single session** - no persistence | Cannot resume conversations | Add database storage |
| **Network dependent** | Poor experience on unstable connections | Connection recovery, offline mode |

---

## Future Work

1. **Vision**: Multi-user dataset, continuous learning
2. **Language**: Larger models, RAG integration
3. **System**: Persistent storage, offline support
4. **Deployment**: Docker containerization, cloud hosting
