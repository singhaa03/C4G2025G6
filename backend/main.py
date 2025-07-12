from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer (make sure the model folder exists and has your model files)
MODEL_PATH = "../model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # Set model to evaluation mode

# Request body format
class TextInput(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict(input: TextInput):
    try:
        # Tokenize the input
        inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)

        # Run prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = round(probs[0][prediction].item(), 4)

        label = "phishing" if prediction == 1 else "not phishing"
        return {"label": label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
