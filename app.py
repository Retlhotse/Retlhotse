from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch
from collections import Counter
import torch.nn.functional as F

# 1️⃣ Load your dataset
dataset = load_dataset("csv", data_files="40k_labeled_stackoverflow_questions_cleaned.csv")

# 2️⃣ Define label mappings first
label2id = {"low": 0, "high": 1}
id2label = {0: "low", 1: "high"}

# 3️⃣ Clean and map labels
def clean_label(example):
    if example["label"] is None:
        print("⚠️ Found missing label:", example)
        return {"label": None}

    label_str = str(example["label"]).strip().lower()

    if label_str not in label2id:
        print(f"⚠️ Found unexpected label: {label_str}")
        return {"label": None}

    return {"label": label2id[label_str]}

# 4️⃣ Apply cleaning
dataset = dataset.map(clean_label)

# 5️⃣ Filter out invalid / missing labels
dataset = dataset.filter(lambda x: x["label"] is not None)

print(dataset)



# 2. Tokenizer
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Tokenize function
def tokenize(batch):
    return tokenizer(batch["question"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)

# 4. Train/val split
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 5. Calculate class weights
labels = train_dataset["label"]
label_counts = Counter(labels)
total_samples = len(labels)
class_weights = [total_samples / (len(label_counts) * label_counts[i]) for i in range(len(label_counts))]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print("Class Weights:", class_weights)

# 6. Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# 7. Override Trainer to use weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 8. Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# 9. Training args
training_args = TrainingArguments(
    output_dir="./modernbert_complexity_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=50
)

# 10. Train
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# 11. Save model
trainer.save_model("./modernbert_complexity_model")
tokenizer.save_pretrained("./modernbert_complexity_model")

# ==============================
# 12. Inference with softmax scores
# ==============================

# Load model for inference
model = AutoModelForSequenceClassification.from_pretrained("./modernbert_complexity_model")
tokenizer = AutoTokenizer.from_pretrained("./modernbert_complexity_model")

def predict_with_confidence(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()

    predicted_class = id2label[int(np.argmax(probs))]
    confidence_scores = {id2label[i]: round(probs[i], 4) for i in range(len(probs))}
    return predicted_class, confidence_scores

# Example
question_text = "How can I implement a distributed hash table from scratch?"
pred_class, conf = predict_with_confidence(question_text)

print("Predicted Complexity:", pred_class)
print("Confidence Scores:", conf)
