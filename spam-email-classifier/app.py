import streamlit as st
import joblib

# Model load
model = joblib.load("models/spam_model.pkl")

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")

st.title("📧 Spam Email Classifier")
st.write("Email ko text paste garnus, ani model le SPAM ho ki haina bhanchha.")

email_text = st.text_area("Email text", height=150, placeholder="Example: You have won a free cash prize")

if st.button("Check"):
    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = model.predict([email_text])[0]
        probs = model.predict_proba([email_text])[0]
        confidence = probs.max() * 100

        if result == "spam":
            st.error(f"🚨 SPAM ({confidence:.1f}% confident)")
        else:
            st.success(f"✅ NOT SPAM ({confidence:.1f}% confident)")