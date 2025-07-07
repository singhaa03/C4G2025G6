import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Spam classifier function
def classify_message(message: str):
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "SPAM" if prediction == 1 else "NOT SPAM"

# List of test messages to classify
test_messages = [
    "Congratulations! You've won a free iPhone. Click here to claim now!",
    "Hey, are we still meeting for coffee tomorrow?",
    "URGENT: Your bank account has been compromised. Act now!",
    "Don't forget to bring the USB drive to class.",
    "You've been selected for a $1000 Walmart gift card.",
    "Bro, you up for the gym later?",
    "Please update your account info to avoid suspension.",
    "Family dinner tonight at 7 — see you there!",
]

# Classify each message
print("📨 Message Classification Results:\n")
for msg in test_messages:
    result = classify_message(msg)
    print(f"\"{msg}\" → {result}")
