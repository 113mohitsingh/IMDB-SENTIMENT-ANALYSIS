import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="IMDB Sentiment App", layout="wide")

st.title("🎬 IMDB Sentiment Analysis")

uploaded_file = st.file_uploader("Upload IMDB Dataset CSV", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Convert sentiment to numeric
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    st.success("Dataset uploaded successfully!")

    st.write("Total Reviews:", len(df))
    st.write("Positive Reviews:", df["sentiment"].sum())
    st.write("Negative Reviews:", len(df) - df["sentiment"].sum())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42
    )

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", max_features=4000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("📊 Model Accuracy")
    st.write(f"Accuracy: {accuracy*100:.2f}%")

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Prediction Section
    st.subheader("🔮 Predict Sentiment")

    user_input = st.text_area("Enter a movie review")

    if st.button("Analyze"):
        if user_input:
            vec = tfidf.transform([user_input])
            prediction = model.predict(vec)[0]
            result = "Positive 😊" if prediction == 1 else "Negative 😞"
            st.success(f"Sentiment: {result}")
        else:
            st.warning("Please enter a review first.")
