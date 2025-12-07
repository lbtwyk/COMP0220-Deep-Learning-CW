# SignTutor Frontend

A premium React-based chat interface for the SignTutor AI assistant, specialized in Sign Language & Deaf Culture education.

## ✨ Features

- **Premium Light Theme** — Pantone-inspired soft palette with beautiful shadows and rounded corners
- **Streaming Text Animation** — Words fade in from transparent to opaque as responses generate
- **Markdown Rendering** — Full support for headers, bold, lists, code blocks, tables
- **Text-to-Speech** — Multiple TTS providers (Browser, Google Cloud, ElevenLabs)
- **Settings Modal** — Configure TTS provider, voice selection, and API keys
- **Responsive Design** — Works on desktop and mobile

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| React 19 | UI framework |
| Vite | Build tool & dev server |
| axios | HTTP client for API calls |
| react-markdown | Markdown rendering |
| remark-gfm | GitHub Flavored Markdown support |

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- Backend running on `http://localhost:8000`

### Install & Run

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open in browser
http://localhost:5173
```

### Or use the root startup script:
```bash
# From project root
./start_app.sh
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── App.jsx        # Main chat component + Settings modal
│   ├── App.css        # All styling (1200+ lines)
│   ├── index.css      # Global styles
│   └── main.jsx       # React entry point
├── index.html         # HTML template
├── package.json       # Dependencies
└── vite.config.js     # Vite configuration
```

## 🎨 UI Components

### Chat Interface
- Welcome screen with example prompts
- User/Assistant message bubbles with avatars
- Typing indicator (bouncing dots)
- Auto-scroll to latest message

### Sidebar
- Logo & branding
- New Conversation button
- Model selector (Cloud AI / Local Qwen3)
- Persona/System prompt editor
- TTS toggle
- Settings gear icon

### Settings Modal
- **TTS Tab**: Provider selection (Browser/Google/ElevenLabs), voice picker
- **API Keys Tab**: Configure OpenAI, ElevenLabs keys

## 🔊 Text-to-Speech

### Supported Providers

| Provider | Setup | Free Tier |
|----------|-------|-----------|
| Browser | None needed | Unlimited |
| Google Cloud | `GOOGLE_APPLICATION_CREDENTIALS` env var | 4M chars/month |
| ElevenLabs | `ELEVENLABS_API_KEY` env var | 10K chars/month |

### Voice Options (ElevenLabs)
- 🥒 Rick Sanchez (Raspy Genius)
- 🎸 Rockstar (Dave Grohl Style)
- Rachel, Bella, Josh, and more...

## 🎯 API Endpoints Used

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send message, receive AI response |
| `/tts` | POST | Convert text to speech audio |
| `/health` | GET | Check backend status |
| `/settings` | GET | Get available providers |

## 📝 Configuration

Settings are stored in `localStorage` under key `signtutor_settings`:

```json
{
  "ttsProvider": "browser",
  "elevenlabsVoice": "ErXwobaYiN019PkySvjV",
  "googleVoice": "en-US-Wavenet-D",
  "openaiKey": "",
  "elevenlabsKey": ""
}
```

## 🏗️ Build for Production

```bash
npm run build
```

Output will be in `dist/` folder, ready for deployment.

## 📜 Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start dev server with HMR |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |

## 🎨 Customization

### Colors
Edit CSS variables in `App.css`:
```css
:root {
  --bg-primary: #FAFBFC;
  --accent-primary: #6366F1;
  /* ... */
}
```

### Fonts
Currently using system fonts (SF Pro on macOS):
```css
font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', ...
```

## 📄 License

Part of the COMP0220 Deep Learning coursework project.
