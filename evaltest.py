# app.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# Load trained model and tokenizer
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def classify_message(message):
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "🚨 SPAM" if prediction == 1 else "✅ Not Spam"

# Gradio Interface
demo = gr.Interface(
    fn=classify_message,
    inputs=gr.Textbox(lines=3, placeholder="Enter a message to classify..."),
    outputs="text",
    title="Spam Message Classifier",
    description="DistilBERT-based model trained on multiple spam datasets to detect spam messages.",
)

if __name__ == "__main__":
    demo.launch()
