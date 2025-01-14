import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

# Load your dataset
# Assuming you have a pandas dataframe 'df' with columns ['cleaned_sentences', 'manual_tagging']
df = pd.read_csv("your_dataset.csv")

# Binarize the manual_tagging column (positive=1, negative=0, neutral=2)
label_mapping = {'positive': 0, 'negative': 1, 'neutral': 2}
df['manual_tagging'] = df['manual_tagging'].map(label_mapping)

# Split data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned_sentences'], df['manual_tagging'], test_size=0.2, random_state=42)

# Tokenization and encoding using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_texts(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=512)

train_encodings = encode_texts(train_texts.tolist())
val_encodings = encode_texts(val_texts.tolist())

# Convert to torch Dataset format
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels.tolist()
})

val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': val_labels.tolist()
})

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Set up Trainer arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)
accuracy = accuracy_score(val_labels, preds)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# If you have an unseen test set, you can use the model to predict on that
# Example for prediction on unseen data
unseen_data = ["This is a great product!", "I had a terrible experience."]
encoded_unseen = tokenizer(unseen_data, padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    model.eval()
    outputs = model(**encoded_unseen)
    predictions = torch.argmax(outputs.logits, dim=1)

print("Predictions for unseen data:", predictions)
