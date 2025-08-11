import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Label mappings (must match training)
label2id = {"low": 0, "high": 1}
id2label = {0: "low", 1: "high"}

# 1️⃣ Load the trained model & tokenizer
model_path = "./modernbert_complexity_model"  # Path to your saved model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure model is in evaluation mode
model.eval()

# 2️⃣ Prediction function with confidence scores
def predict_with_confidence(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()

    predicted_class = id2label[int(np.argmax(probs))]
    confidence_scores = {id2label[i]: round(probs[i], 4) for i in range(len(probs))}
    return predicted_class, confidence_scores

# 3️⃣ Example usage
if __name__ == "__main__":
    question_text = "How can I implement a distributed hash table from scratch?"
    pred_class, conf = predict_with_confidence(question_text)

    print(f"Predicted Complexity: {pred_class}")
    print(f"Confidence Scores: {conf}")
