# api/main.py

from emotion_detector import detect_emotion
from fastapi import FastAPI
from pydantic import BaseModel
from response_generator import generate_support_message

app = FastAPI(name="Moodmate API", version="1.0.0")


# -------- Request Schemas --------


class EmotionRequest(BaseModel):
    text: str


class SupportRequest(BaseModel):
    text: str
    emotion: str


# -------- Endpoints --------


@app.post("/analyze_emotion")
def analyze_emotion(request: EmotionRequest):
    emotion = detect_emotion(request.text)
    return {"emotion": emotion}


@app.post("/generate_support")
def generate_support(request: SupportRequest):
    message = generate_support_message(request.text, request.emotion)
    return {"support_message": message}


@app.get("/")
def health_check():
    return {"status": "MoodMate API is running"}


# -------- End of File --------
