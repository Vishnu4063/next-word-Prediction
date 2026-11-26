import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the trained model and prediction function directly from nxtprediction.py
from nxtprediction import predict_next_word, tokenizer

app = FastAPI(
    title="Next Word Prediction API",
    description="FastAPI connecting the LSTM model from nxtprediction.py with an interactive frontend.",
    version="1.0.0"
)

# Enable CORS for frontend flexibility (servable from any origin or local file)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "index.html"


class PredictionRequest(BaseModel):
    text: str = Field(..., description="Input seed text", examples=["I love"])
    num_words: int = Field(1, ge=1, le=10, description="Number of sequential words to predict")


class PredictionResponse(BaseModel):
    input_text: str
    predicted_word: Optional[str] = None
    predicted_words: List[str] = []
    result: Optional[str] = None
    success: bool
    message: str


def generate_words(seed_text: str, num_words: int = 1):
    cleaned = seed_text.strip()
    if not cleaned:
        return {
            "input_text": seed_text,
            "predicted_word": None,
            "predicted_words": [],
            "result": seed_text,
            "success": False,
            "message": "Input text cannot be empty."
        }

    current_text = cleaned
    predicted_words = []

    for _ in range(num_words):
        next_word = predict_next_word(current_text)
        if not next_word:
            break
        predicted_words.append(next_word)
        current_text = f"{current_text} {next_word}"

    if not predicted_words:
        return {
            "input_text": seed_text,
            "predicted_word": None,
            "predicted_words": [],
            "result": seed_text,
            "success": False,
            "message": f"Could not predict next word (words in '{seed_text}' are not in vocabulary)."
        }

    return {
        "input_text": seed_text,
        "predicted_word": predicted_words[0],
        "predicted_words": predicted_words,
        "result": current_text,
        "success": True,
        "message": "Prediction successful."
    }


@app.get("/", tags=["Frontend"])
def serve_frontend():
    if INDEX_HTML_PATH.exists():
        return FileResponse(INDEX_HTML_PATH, media_type="text/html")
    return JSONResponse(
        status_code=404,
        content={"error": "index.html not found. Please ensure index.html exists in the project root."}
    )


@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "online",
        "model": "LSTM (nxtprediction.py)",
        "vocabulary_size": len(tokenizer.word_index)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_endpoint(request: PredictionRequest):
    result = generate_words(request.text, request.num_words)
    return PredictionResponse(**result)


@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_get_endpoint(
    text: str = Query(..., description="Input seed text"),
    num_words: int = Query(1, ge=1, le=10, description="Number of sequential words to predict")
):
    result = generate_words(text, num_words)
    return PredictionResponse(**result)


@app.get("/vocabulary", tags=["Information"])
def get_vocabulary():
    return {
        "total_words": len(tokenizer.word_index),
        "vocabulary": tokenizer.word_index
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
