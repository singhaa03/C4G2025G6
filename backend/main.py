from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Allow all origins for development (OK for local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow everything (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # set model to eval mode

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the input schema
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item() * 100

    label = "Phishing" if prediction == 1 else "Not Phishing"

    return {
        "prediction": label,  # 👈 this must exist
        "confidence": round(confidence, 2)
    }

