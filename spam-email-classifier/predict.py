import joblib

model = joblib.load("models/spam_model.pkl")

while True:
    text = input("\nEmail text (q to quit): ")
    if text.lower() == "q":
        break
    result = model.predict([text])[0]
    prob = model.predict_proba([text]).max() * 100
    print(f"{'SPAM' if result == 'spam' else 'NOT SPAM'} ({prob:.1f}% confident)")