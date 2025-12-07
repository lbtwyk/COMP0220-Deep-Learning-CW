#!/bin/bash

# Load API keys from file (keeps secrets out of git)
if [ -f "openai_api_key" ]; then
    source openai_api_key
    echo "Loaded OpenAI API key from file"
fi

# Google TTS credentials
export GOOGLE_APPLICATION_CREDENTIALS="/home/hz/Downloads/comp0220-25f533457317.json"

# Default to asking for API Key if not present
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY is not set."
    echo "To use the API mode, please export your key first:"
    echo "export OPENAI_API_KEY='your-key-here'"
    echo "Or continue to run only local mode (if dependencies exist)."
fi

echo "Starting Backend..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "Backend running on PID $BACKEND_PID"
echo "Frontend running on PID $FRONTEND_PID"

trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT SIGTERM

wait
