# -*- coding: utf-8 -*-
"""
Fine-tunes a RoBERTa model for classifying question complexity and allows for
separate prediction using the saved model.

This script is divided into two main parts:
1.  A `train_and_save_model` function that fine-tunes and saves the model.
2.  A `predict_complexity` function that loads the saved model to make predictions
    and show confidence scores.
"""

# First, ensure you have the necessary libraries installed.
# You can install them using pip:
# pip install torch transformers pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import os


def train_and_save_model():
    """
    Main function to run the RoBERTa fine-tuning and saving process.
    """
    # --- 1. Load and Prepare the Dataset ---
    try:
        # Load the dataset from the specified CSV file
        df = pd.read_csv('low complexity.csv')
        print("Successfully loaded 'low complexity.csv'")
        print("Dataset preview:")
        print(df.head())
    except FileNotFoundError:
        print("Error: 'low complexity.csv' not found.")
        print("Please ensure the CSV file is in the same directory as this script.")
        # Create a dummy dataframe for demonstration purposes if the file is not found
        print("Creating a dummy dataframe to proceed with the script execution.")
        data = {
            'question': [
                'How do I declare a variable in Python?',
                'What is the difference between a list and a tuple?',
                'Explain the concept of recursion.',
                'How does asynchronous programming work in JavaScript?',
                'Implement a binary search tree from scratch.',
                'What are the principles of SOLID design?',
                'How to reverse a string in Python?',
                'What is a decorator in Python?',
                'Explain time complexity for a sorting algorithm.',
                'How to handle memory management in C++?'
            ],
            'label': ['low', 'low', 'high', 'high', 'high', 'high', 'low', 'low', 'high', 'high']
        }
        df = pd.DataFrame(data)

    # Map string labels to integer IDs
    label_map = {'low': 0, 'high': 1}
    df['label'] = df['label'].map(label_map)

    # Separate features (questions) and labels
    questions = df['question'].tolist()
    labels = df['label'].tolist()

    # Split the dataset into training and testing sets (80% train, 20% test)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        questions, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # --- 2. Tokenization ---
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    # --- 3. Create a PyTorch Dataset ---
    class ComplexityDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ComplexityDataset(train_encodings, train_labels)
    val_dataset = ComplexityDataset(val_encodings, val_labels)

    # --- 4. Define the Model ---
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    # --- 5. Define a Compute Metrics Function ---
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    # --- 6. Set Training Arguments ---
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # --- 7. Initialize the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # --- 8. Train the Model ---
    print("\nStarting model training...")
    trainer.train()
    print("Training finished.\n")

    # --- 9. Evaluate and Save ---
    print("Evaluating the final model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # The trainer automatically saves the best model checkpoints in the output_dir.
    # We can also explicitly save the final model.
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Model and tokenizer saved successfully.")


def predict_complexity(question_text, model_dir='./results'):
    """
    Loads a fine-tuned model from a directory and predicts the complexity
    of a given question, including the confidence score.

    Args:
        question_text (str): The question to classify.
        model_dir (str): The directory where the fine-tuned model is saved.

    Returns:
        tuple: A tuple containing the predicted label (str) and confidence score (float).
    """
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        print("Please run the training process first by calling the `train_and_save_model()` function.")
        return None, None

    try:
        # Load the tokenizer and model from the saved directory
        tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        model = RobertaForSequenceClassification.from_pretrained(model_dir)
        print(f"\n--- Loading model from {model_dir} for prediction ---")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

    print("Running prediction...")
    inputs = tokenizer(question_text, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to the same device as the model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply softmax to logits to get probabilities (confidence scores)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get the prediction and its corresponding confidence score
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence_score = probabilities[0][prediction].item()

    predicted_label = 'high' if prediction == 1 else 'low'

    print(f"\nThe question: '{question_text}'")
    print(f"Predicted Complexity: '{predicted_label}'")
    print(f"Confidence Score: {confidence_score:.2%}")  # Format as a percentage

    return predicted_label, confidence_score


if __name__ == '__main__':
    # --- STEP 1: Train and save the model (RUN THIS ONCE) ---
    # To train your model, uncomment the line below and run the script.
    # Once training is complete, a 'results' folder will be created with your model.
    # train_and_save_model()

    # --- STEP 2: Predict using the saved model (RUN THIS ANYTIME) ---
    # After the model is trained and saved, you can comment out the training line
    # and use the `predict_complexity` function to classify new questions.
    print("\n--- Starting prediction mode ---")

    # Example questions to test:
    # question1 = "Can you build a BERT model for me?"
    # question2 = "What is a variable?"
    question = ""
    while question.lower() != "quit":
        question = input("Ask your question (or type 'quit' to exit): ")
        if question.lower() != "quit" and question.strip() != "":
            predict_complexity(question)