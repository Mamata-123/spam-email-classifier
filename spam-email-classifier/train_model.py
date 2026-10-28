
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# 1. Load dataset
data = pd.read_csv("data/emails.csv")

# 2. Separate email text and labels
X = data["text"]
y = data["label"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Create the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# 5. Train the model
model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== SPAM EMAIL CLASSIFIER =====")
print(f"Total emails: {len(data)}")
print(f"Training emails: {len(X_train)}")
print(f"Testing emails: {len(X_test)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 7. Predict a new email
new_email = [
    "Congratulations! You won a free prize. Click now!"
]

result = model.predict(new_email)[0]

print("\nNew Email Prediction:")
print("SPAM" if result == "spam" else "NOT SPAM")


import joblib

joblib.dump(model, "models/spam_model.pkl")
print("\nModel saved to models/spam_model.pkl")