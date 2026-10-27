import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Assign labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real], ignore_index=True)
data = data[["text", "label"]]

# Remove missing and empty text
data.dropna(subset=["text"], inplace=True)
data = data[data["text"].str.strip() != ""]

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["clean_text"] = data["text"].apply(clean_text)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Data preprocessing completed successfully!")
print("Total articles:", len(data))
print("Training samples:", X_train_tfidf.shape[0])
print("Testing samples:", X_test_tfidf.shape[0])
print("TF-IDF features:", X_train_tfidf.shape[1])
print("\nOriginal text:", data["text"].iloc[0][:200])
print("\nCleaned text:", data["clean_text"].iloc[0][:200])