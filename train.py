import torch
from datasets import load_dataset, concatenate_datasets, Features, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Ensure directories exist
os.makedirs("./model", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Load the tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def clean_dataset(ds, text_key="text", label_key="label", spam_value="spam"):
    """Standardizes the dataset (handles label encoding and column selection)."""
    def standardize(example):
        label = example[label_key]
        if isinstance(label, str):
            label = 1 if label == spam_value else 0
        elif isinstance(label, int):
            label = label
        else:
            label = 0
        return {"text": example[text_key], "label": int(label)}

    cleaned = ds.map(standardize)
    cleaned = cleaned.remove_columns([col for col in cleaned.column_names if col not in ["text", "label"]])

    features = Features({
        "text": Value("string"),
        "label": Value("int64"),
    })
    cleaned = cleaned.cast(features)
    return cleaned

# Load datasets with caching for faster reloading
sms = clean_dataset(load_dataset("sms_spam", split="train", cache_dir="./cache"), text_key="sms", label_key="label")
enron = clean_dataset(load_dataset("SetFit/enron_spam", split="train", cache_dir="./cache"), text_key="text", label_key="label", spam_value="spam")
more_spam = clean_dataset(load_dataset("Deysi/spam-detection-dataset", split="train", cache_dir="./cache"), text_key="text", label_key="label")

# Combine the datasets
combined = concatenate_datasets([sms, enron, more_spam]).shuffle(seed=24)

combined = combined.select(range(1000))



# Tokenization with parallel processing (using multiple cores to speed up tokenization)
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# Apply tokenization with parallelization
tokenized_dataset = combined.map(tokenize_function, batched=True, num_proc=4)  # Parallel processing
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# Split dataset into train and test
split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
eval_dataset = split["test"]

# Load the model and move it to the GPU if available
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Define the evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training Arguments (with eval_strategy renamed to eval_strategy for compatibility)
training_args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",  # ✔ evaluates at the end of each epoch
    save_strategy="epoch",  # ✔ saves model every epoch
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    load_best_model_at_end=True,  # ✔ loads the best model at the end of training
    logging_dir="./logs",
    logging_steps=50,
    
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Start the training process
trainer.train(resume_from_checkpoint=True)
trainer.save_model("./model")
tokenizer.save_pretrained("./model")

final_results = trainer.evaluate()
print("🔍 Final Evaluation Metrics:")
for key, value in final_results.items():
    print(f"{key}: {value:.4f}")
