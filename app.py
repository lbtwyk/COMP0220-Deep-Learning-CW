import os
import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path

# Try to import API clients
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from elevenlabs import ElevenLabs
except ImportError:
    ElevenLabs = None

try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False

# For Google TTS via API key (simpler setup)
import requests
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")

# Import local inference logic (optional)
try:
    from inference import load_finetuned_model, generate_response
    LOCAL_INFERENCE_AVAILABLE = True
except ImportError:
    LOCAL_INFERENCE_AVAILABLE = False
    print("Warning: Local inference dependencies not found. Only external API will work.")

app = FastAPI(title="Qwen3 Tutor API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
tokenizer = None
device = None
openai_client = None
elevenlabs_client = None
google_tts_client = None

# Initialize clients
if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if os.getenv("ELEVENLABS_API_KEY") and ElevenLabs:
    elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

if GOOGLE_TTS_AVAILABLE and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    google_tts_client = texttospeech.TextToSpeechClient()

# ============== Chat Models ==============

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a friendly tutor who explains sign languages and Deaf culture clearly."
    temperature: float = 0.7
    max_new_tokens: int = 512
    use_api: bool = True
    api_model: str = "gpt-3.5-turbo"

class ChatResponse(BaseModel):
    response: str
    source: str

# ============== TTS Models ==============

class TTSRequest(BaseModel):
    text: str
    provider: str = "google"  # "google", "elevenlabs", "browser"
    voice_id: Optional[str] = None
    # Google TTS options
    language_code: str = "en-US"
    # ElevenLabs options
    model_id: str = "eleven_monolingual_v1"

class TTSVoice(BaseModel):
    id: str
    name: str
    provider: str
    language: Optional[str] = None
    preview_url: Optional[str] = None

class TTSVoicesResponse(BaseModel):
    voices: List[TTSVoice]

# ============== Startup ==============

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, device
    if LOCAL_INFERENCE_AVAILABLE:
        print("Local inference modules available.")
    else:
        print("Local inference not available.")
    
    print(f"OpenAI configured: {openai_client is not None}")
    print(f"ElevenLabs configured: {elevenlabs_client is not None}")
    print(f"Google TTS configured: {google_tts_client is not None or GOOGLE_TTS_API_KEY is not None}")

# ============== Chat Endpoints ==============

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global model, tokenizer, device, openai_client

    if request.use_api:
        if not openai_client:
            if os.getenv("OPENAI_API_KEY"):
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            else:
                return ChatResponse(
                    response="Error: OPENAI_API_KEY not found. Please set it in environment variables.",
                    source="error"
                )
        
        try:
            completion = openai_client.chat.completions.create(
                model=request.api_model,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.message}
                ],
                temperature=request.temperature,
                max_tokens=request.max_new_tokens
            )
            return ChatResponse(
                response=completion.choices[0].message.content,
                source="api"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

    # Local model
    if not LOCAL_INFERENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Local inference not available.")

    if model is None:
        print("Loading local model...")
        finetuned_path = Path("./qwen3_finetuned/final")
        base_model = "Qwen/Qwen3-4B-Instruct-2507"
        try:
            model_path = str(finetuned_path) if finetuned_path.exists() else base_model
            model, tokenizer, device = load_finetuned_model(
                model_path=model_path,
                base_model=base_model,
                device="auto"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        response_text = generate_response(
            model=model,
            tokenizer=tokenizer,
            device=device,
            user_prompt=request.message,
            system_prompt=request.system_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )
        return ChatResponse(response=response_text, source="local")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============== TTS Endpoints ==============

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech from text using the specified provider."""
    
    if request.provider == "google":
        # Method 1: Use Service Account JSON (preferred - set via GOOGLE_APPLICATION_CREDENTIALS)
        if google_tts_client:
            try:
                synthesis_input = texttospeech.SynthesisInput(text=request.text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code=request.language_code,
                    name=request.voice_id or f"{request.language_code}-Wavenet-D"
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                response = google_tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                return StreamingResponse(
                    io.BytesIO(response.audio_content),
                    media_type="audio/mpeg",
                    headers={"Content-Disposition": "inline; filename=speech.mp3"}
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Google TTS Error: {str(e)}")
        
        else:
            raise HTTPException(status_code=503, detail="Google TTS not configured. Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON file.")
    
    elif request.provider == "elevenlabs":
        if not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs not configured. Set ELEVENLABS_API_KEY.")
        
        try:
            audio = elevenlabs_client.generate(
                text=request.text,
                voice=request.voice_id or "Rachel",
                model=request.model_id
            )
            
            # Collect audio bytes
            audio_bytes = b"".join(audio)
            
            return StreamingResponse(
                io.BytesIO(audio_bytes),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline; filename=speech.mp3"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ElevenLabs Error: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown TTS provider: {request.provider}")

@app.get("/tts/voices", response_model=TTSVoicesResponse)
async def get_tts_voices(provider: str = "google"):
    """Get available voices for a TTS provider."""
    
    voices = []
    
    if provider == "google":
        if not google_tts_client:
            raise HTTPException(status_code=503, detail="Google TTS not configured.")
        
        try:
            response = google_tts_client.list_voices()
            for voice in response.voices:
                if voice.language_codes[0].startswith("en"):
                    voices.append(TTSVoice(
                        id=voice.name,
                        name=voice.name,
                        provider="google",
                        language=voice.language_codes[0]
                    ))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    elif provider == "elevenlabs":
        if not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs not configured.")
        
        try:
            response = elevenlabs_client.voices.get_all()
            for voice in response.voices:
                voices.append(TTSVoice(
                    id=voice.voice_id,
                    name=voice.name,
                    provider="elevenlabs",
                    preview_url=voice.preview_url
                ))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return TTSVoicesResponse(voices=voices)

# ============== Health & Status ==============

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "local_model_loaded": model is not None,
        "openai_configured": openai_client is not None,
        "elevenlabs_configured": elevenlabs_client is not None,
        "google_tts_configured": google_tts_client is not None or GOOGLE_TTS_API_KEY is not None
    }

@app.get("/settings")
async def get_settings():
    """Return current configuration status for the settings page."""
    return {
        "chat": {
            "openai_available": openai_client is not None,
            "local_available": LOCAL_INFERENCE_AVAILABLE
        },
        "tts": {
            "browser": True,  # Always available
            "google": google_tts_client is not None or GOOGLE_TTS_API_KEY is not None,
            "elevenlabs": elevenlabs_client is not None
        }
    }
