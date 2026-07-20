import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load stopwords
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# App Title
st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("Enter a news article below and let the AI predict whether it is Fake or Real.")

news = st.text_area(
    "Paste News Article",
    height=250
)

if st.button("🔍 Check News"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max() * 100

        if prediction == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")

        st.write(f"**Confidence:** {confidence:.2f}%")