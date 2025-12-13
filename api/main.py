from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from fastapi.responses import RedirectResponse
from emotion_detector import EmotionDetector, model
from response_generator import ResponseGenerator


app = FastAPI(name="Moodmate API", version="1.0.0")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok",
            "model_loaded": model is not None}


class MoodRequest(BaseModel):
    text: str


@app.post("/predict", tags=["Mood Prediction"])
async def predict_mood(request: MoodRequest):
    text = request.text
    emotion = EmotionDetector(text)
    response = ResponseGenerator(emotion)
    return {"emotion": emotion, "response": response} 


@app.post("/generate_response", tags=["Response Generation"])
async def generate_response(request: MoodRequest):
    text = request.text
    response = ResponseGenerator(text)
    return {"response": response}