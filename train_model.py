import pandas as pd
import re
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)
data = data.dropna(subset=["text"])

data["text"] = data["text"].astype(str)

# Step 2: Clean the news text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["clean_text"] = data["text"].apply(clean_text)
data = data[data["clean_text"] != ""]

print("Dataset loaded successfully!")
print("Total articles:", len(data))
print("Fake articles:", (data["label"] == 0).sum())
print("Real articles:", (data["label"] == 1).sum())

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# Step 4: Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nText preprocessing completed!")
print("Training samples:", X_train_tfidf.shape[0])
print("Testing samples:", X_test_tfidf.shape[0])
print("TF-IDF features:", X_train_tfidf.shape[1])

# Step 5: Train the machine learning algorithm
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluate the model
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL EVALUATION RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["Fake News", "Real News"]
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Step 7: Save the trained model
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")

# Step 8: Test sample news
def predict_news(news):
    cleaned = clean_text(news)
    features = vectorizer.transform([cleaned])
    result = model.predict(features)[0]
    return "Real News" if result == 1 else "Fake News"

sample = input("\nEnter a news article to test: ")
print("Prediction:", predict_news(sample))