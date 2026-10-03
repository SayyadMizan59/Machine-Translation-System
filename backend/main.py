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
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Check candidate directories in order of priority:
# 1. Environment variable MODEL_PATH (if set)
# 2. Inside backend directory: backend/final_english_hindi_model
# 3. In project root: final_english_hindi_model
# 4. Relative to current working directory
candidates = [
    Path(os.environ["MODEL_PATH"]).resolve() if "MODEL_PATH" in os.environ else None,
    CURRENT_DIR / "final_english_hindi_model",
    PROJECT_ROOT / "final_english_hindi_model",
    Path("./backend/final_english_hindi_model").resolve(),
    Path("./final_english_hindi_model").resolve(),
]

MODEL_DIR = None
for candidate in candidates:
    if candidate and candidate.exists() and (candidate / "config.json").exists():
        MODEL_DIR = candidate
        break

if MODEL_DIR is None:
    # Default fallback
    MODEL_DIR = (CURRENT_DIR / "final_english_hindi_model") if (CURRENT_DIR / "final_english_hindi_model").exists() else (PROJECT_ROOT / "final_english_hindi_model")

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
    print(f"[*] Resolved model directory: {MODEL_DIR}...")

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
# Default local origins for development
default_origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

# Allow custom frontend URL(s) from environment variables (comma-separated if multiple)
env_origins = os.getenv("ALLOWED_ORIGINS", os.getenv("FRONTEND_URL", ""))
if env_origins:
    custom_origins = [orig.strip() for orig in env_origins.split(",") if orig.strip()]
    allowed_origins = list(set(default_origins + custom_origins))
else:
    allowed_origins = default_origins

if os.getenv("ALLOW_ALL_ORIGINS", "false").lower() in ("true", "1"):
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https:\/\/.*\.onrender\.com$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
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

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
