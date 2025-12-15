# COMP0220 Deep Learning Coursework
# AI-Powered Educational Podcast System with ASL Vision Input

A full-stack application that combines deep learning vision models, fine-tuned language models, and modern web technologies to create an interactive AI podcast system. Users can type questions or use American Sign Language (ASL) gestures to interact with AI hosts who engage in natural podcast-style conversations.

## 🎯 Project Overview

This project demonstrates the integration of multiple deep learning components:

- **Vision System**: YOLO hand detection + ResNet18 ASL letter recognition
- **Language Model**: Fine-tuned Qwen3-4B for podcast-style dialogue generation
- **Frontend**: React-based modern web interface with real-time WebSocket communication
- **Backend**: FastAPI server orchestrating all components

## 📁 Project Structure

```
COMP0220-Deep-Learning-CW/
├── app.py                      # Main FastAPI backend server
├── start_app.sh                # Quick start script for both frontend + backend
├── requirements.txt            # Backend/inference dependencies
├── requirements_training.txt   # Model training dependencies
│
├── frontend/                   # React web application
│   ├── src/
│   │   ├── components/
│   │   │   ├── PodcastView.jsx      # Main podcast interface
│   │   │   ├── VisionSidebar.jsx    # ASL vision input component
│   │   │   └── ...
│   │   └── App.jsx
│   └── package.json
│
├── podcast/                    # Podcast generation system
│   ├── agents/                 # AI host characters
│   │   ├── base.py             # Base agent class
│   │   ├── rick.py             # "Rick" host persona
│   │   ├── morty.py            # "Morty" co-host persona
│   │   └── summer.py           # "Summer" educational host
│   ├── services/
│   │   ├── state_machine.py    # Conversation flow controller
│   │   ├── podcast_generator.py # Multi-turn dialogue generator
│   │   └── local_model.py      # Local Qwen3 inference
│   └── prompts/                # System prompts for each host
│
├── vision/                     # Computer vision system
│   ├── ASL/
│   │   ├── training/           # Model training scripts
│   │   ├── inference/          # Real-time recognition scripts
│   │   └── models/             # Trained model checkpoints
│   ├── yolo/                   # YOLO hand detection
│   │   └── ckpt/               # YOLO weights
│   └── services/
│       ├── letter_recognizer.py # ASL recognition service
│       └── word_builder.py      # Letter-to-word buffer
│
└── qwen3_finetuning/           # Language model fine-tuning
    ├── finetune_qwen3.py       # Training script
    └── merged_podast_style_data.json  # Training data
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (recommended for inference)
- Conda (for environment management)

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd COMP0220-Deep-Learning-CW
conda create -n comp0220 python=3.10
conda activate comp0220
```

2. **Install backend dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies:**
```bash
cd frontend
npm install
cd ..
```

4. **Start the application:**
```bash
./start_app.sh
```

This starts:
- Backend API at `http://localhost:8000`
- Frontend at `http://localhost:5173`

## 🧠 Core Components

### Vision System (YOLO + ResNet18)

The vision pipeline enables ASL gesture input:

| Component | Model | Purpose |
|-----------|-------|---------|
| Hand Detection | YOLOv11 | Localizes hands in webcam frames |
| Letter Recognition | ResNet18 | Classifies hand poses into A-Y letters |

**Key Files:**
- `vision/services/letter_recognizer.py` - Main recognition service
- `vision/services/word_builder.py` - Buffers letters into words
- `vision/ASL/training/train_from_collected_data.py` - Training script

**Architecture:**
```
Webcam → YOLO (hand bbox) → Crop & Resize → ResNet18 → Letter (A-Y)
                                                    ↓
                                            Word Buffer → Podcast Input
```

### Language Model (Qwen3-4B Fine-tuned)

Custom fine-tuned model for podcast-style dialogue:

- **Base Model**: Qwen/Qwen3-4B
- **Training Method**: LoRA fine-tuning with TRL
- **Training Data**: Educational dialogue dataset (~500 samples)
- **Output**: Natural multi-turn conversations

**Key Files:**
- `qwen3_finetuning/finetune_qwen3.py` - Training script
- `podcast/services/local_model.py` - Inference wrapper

**Hyperparameters:**
```python
# Training Configuration
learning_rate = 2e-4
batch_size = 4
gradient_accumulation = 4
max_seq_length = 2048
lora_r = 16
lora_alpha = 32
num_epochs = 3
```

### Frontend (React + Vite)

Modern web interface with:
- Real-time podcast streaming via WebSocket
- Vision sidebar for ASL input
- Text-to-Speech integration (Google TTS, ElevenLabs)
- Responsive iOS-style design

**Key Files:**
- `frontend/src/components/PodcastView.jsx` - Main podcast UI
- `frontend/src/components/VisionSidebar.jsx` - Vision input
- `frontend/src/hooks/useVisionWebSocket.js` - Vision WebSocket

### Backend (FastAPI)

RESTful API + WebSocket endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /podcast/start` | Start new podcast session |
| `WS /ws/podcast/{id}` | Real-time podcast streaming |
| `WS /ws/vision` | Vision input streaming |
| `GET /health` | Health check |

**Key Files:**
- `app.py` - Main server with all endpoints
- `podcast/services/state_machine.py` - Conversation flow

## 📊 Model Training

### Vision Model

```bash
# From vision/ASL/training/
python train_from_collected_data.py
```

### Language Model

```bash
# Install training dependencies
pip install -r requirements_training.txt

# Run fine-tuning
cd qwen3_finetuning
python finetune_qwen3.py
```

## 🔧 Configuration

### Environment Variables

```bash
# Optional API keys (for cloud TTS)
export OPENAI_API_KEY="your-key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

### Vision Settings

In `app.py`:
```python
word_builder = WordBuilder(
    debounce_threshold=0.3,    # Seconds between same letter
    confidence_threshold=0.4   # Minimum confidence for letter
)
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [PODCAST_ARCHITECTURE.md](PODCAST_ARCHITECTURE.md) | Detailed podcast system design |
| [PODCAST_BACKEND_GUIDE.md](PODCAST_BACKEND_GUIDE.md) | Backend API documentation |
| [vision/VISION_INTEGRATION_GUIDE.md](vision/VISION_INTEGRATION_GUIDE.md) | Vision system details |
| [CONTRIBUTIONS.md](CONTRIBUTIONS.md) | Team contributions |
| [LIMITATIONS.md](LIMITATIONS.md) | Known limitations |

## 🎬 Demo

To record a demo video:
1. Start the application with `./start_app.sh`
2. Open `http://localhost:5173`
3. Use the podcast interface or enable vision input
4. Record your interaction

See [DEMO_VIDEO_SCRIPT.md](DEMO_VIDEO_SCRIPT.md) for demo guidelines.

## 📄 License

This project is submitted as coursework for COMP0220 Deep Learning at UCL.

## 👥 Team

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for detailed team contributions.
