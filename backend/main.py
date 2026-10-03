import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------
# Configuration and Path Resolution
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "final_english_hindi_model"
if not MODEL_DIR.exists():
    MODEL_DIR = Path("./final_english_hindi_model").resolve()

# Global model state
tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForSeq2SeqLM] = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Model metadata constants
MODEL_METADATA = {
    "model_type": "MarianMTModel",
    "translation_direction": "English to Hindi",
    "training_examples": 20000,
    "epochs": 2,
    "training_loss": 2.108107,
    "validation_loss": 3.856888,
    "bleu_score": 12.05,
    "device": device,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model and tokenizer once when the FastAPI server starts.
    """
    global tokenizer, model, device
    print(f"[*] Initializing model on device: {device}...")
    print(f"[*] Loading model from: {MODEL_DIR}...")

    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model directory not found at: {MODEL_DIR}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
        model.to(device)
        model.eval()
        print("[✓] Model and tokenizer loaded successfully into memory.")
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        raise e

    yield

    # Clean up resources on shutdown if needed
    print("[*] Shutting down translation service...")


# ---------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="English to Hindi Machine Translation API",
    description="High-performance English to Hindi translation backend powered by a custom MarianMT model.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request and Response Schemas
# ---------------------------------------------------------
class TranslationRequest(BaseModel):
    text: str = Field(..., description="English input text to translate", example="How are you?")


class TranslationResponse(BaseModel):
    input: str
    translation: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_type: str
    translation_direction: str
    training_examples: int
    epochs: int
    training_loss: float
    validation_loss: float
    bleu_score: float
    device: str


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint returning system status.
    """
    return {"status": "ok"}


@app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate_text(payload: TranslationRequest):
    """
    Translates English input text into Hindi using the loaded MarianMT model.
    """
    global tokenizer, model, device

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")

    text = payload.text.strip()
    if not text:
        return {"input": payload.text, "translation": ""}

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )

        translation = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return {
            "input": payload.text,
            "translation": translation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
async def get_model_info():
    """
    Returns training and architecture metadata for the translation model.
    """
    info = dict(MODEL_METADATA)
    info["device"] = device
    return info


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
