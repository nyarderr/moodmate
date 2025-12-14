# api/main.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .emotion_detector import detect_emotion
from .response_generator import generate_support_message

app = FastAPI(
    name="Moodmate API",
    version="1.0.0",
    title="Moodmate API",
    description="Emotion-aware NLP microservice for mental wellness support.",
)

# -------- Request Schemas --------


class EmotionRequest(BaseModel):
    text: str


class SupportRequest(BaseModel):
    text: str
    emotion: str


# -------- Endpoints --------


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


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


@app.post("/analyze_and_support)")
def analyze_and_support(request: EmotionRequest):
    emotion = detect_emotion(request.text)
    message = generate_support_message(request.text, emotion)
    return {"emotion": emotion, "support_message": message}


# -------- End of File --------
