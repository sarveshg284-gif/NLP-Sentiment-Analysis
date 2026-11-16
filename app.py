
import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="NLP Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# ---------------------------------
# Custom Styling
# ---------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 700;
}
.subtitle {
    color: #8b8b8b;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Dataset and Train Models
# ---------------------------------
@st.cache_resource
def train_models():

    df = pd.read_csv("reviews.csv")

    required = {"review_text", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Dataset must contain review_text and sentiment columns."
        )

    df = df.dropna(subset=["review_text", "sentiment"]).copy()
    df["review_text"] = df["review_text"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    stop_words = set(stopwords.words("english"))
    stop_words -= {"not", "no", "nor"}

    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = re.sub(r"<.*?>", "", str(text))
        text = text.lower()
        words = re.findall(r"\b[a-z]+\b", text)

        cleaned_words = []

        for word in words:
            if word not in stop_words:
                cleaned_words.append(
                    lemmatizer.lemmatize(word)
                )

        return " ".join(cleaned_words)

    df["cleaned_review"] = df["review_text"].apply(
        preprocess_text
    )

    if df["sentiment"].nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["cleaned_review"])
    y = df["sentiment"]

    nb_model = MultinomialNB()
    nb_model.fit(X, y)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X, y)

    return df, vectorizer, nb_model, lr_model, preprocess_text


# ---------------------------------
# Header
# ---------------------------------
st.markdown(
    '<div class="main-title">💬 NLP Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze customer reviews using Machine Learning</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------
# Load Models
# ---------------------------------
try:
    df, vectorizer, nb_model, lr_model, preprocess_text = train_models()
except Exception as error:
    st.error(f"Unable to load or train models: {error}")
    st.stop()

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("📌 Project Information")
st.sidebar.write("Natural Language Processing")
st.sidebar.write("TF-IDF Feature Extraction")
st.sidebar.write("Multinomial Naive Bayes")
st.sidebar.write("Logistic Regression")

st.sidebar.divider()
st.sidebar.metric("Dataset Reviews", len(df))
st.sidebar.metric("Sentiment Classes", df["sentiment"].nunique())

# ---------------------------------
# Dashboard Metrics
# ---------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Reviews", len(df))
col2.metric("ML Algorithms", 2)
col3.metric("Sentiment Categories", df["sentiment"].nunique())

st.divider()

# ---------------------------------
# Sentiment Distribution
# ---------------------------------
st.subheader("📊 Dataset Sentiment Distribution")

distribution = df["sentiment"].value_counts()
st.bar_chart(distribution)

# ---------------------------------
# Prediction Section
# ---------------------------------
st.subheader("🔍 Analyze Your Review")

review = st.text_area(
    "Enter a customer review:",
    placeholder="Example: The product quality is excellent and delivery was fast!",
    height=130
)

selected_model = st.selectbox(
    "Choose Machine Learning Model",
    ["Logistic Regression", "Naive Bayes", "Compare Both"]
)

if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():
        st.warning("Please enter a customer review.")
    else:
        cleaned = preprocess_text(review)

        if not cleaned.strip():
            st.warning("Please enter a review containing English words.")
        else:
            vector = vectorizer.transform([cleaned])

            st.divider()
            st.subheader("📈 Prediction Results")

            models = {
                "Logistic Regression": lr_model,
                "Naive Bayes": nb_model
            }

            selected = (
                models.items()
                if selected_model == "Compare Both"
                else [(selected_model, models[selected_model])]
            )

            result_cols = st.columns(len(list(selected)))

            selected = (
                models.items()
                if selected_model == "Compare Both"
                else [(selected_model, models[selected_model])]
            )

            for index, (name, model) in enumerate(selected):

                prediction = model.predict(vector)[0]
                probabilities = model.predict_proba(vector)[0]
                confidence = max(probabilities) * 100

                with result_cols[index]:

                    st.markdown(f"### {name}")

                    if prediction == "positive":
                        st.success("😊 Positive Sentiment")
                    elif prediction == "negative":
                        st.error("😞 Negative Sentiment")
                    elif prediction == "neutral":
                        st.info("😐 Neutral Sentiment")
                    else:
                        st.info(f"Predicted: {prediction.title()}")

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    probability_df = pd.DataFrame({
                        "Sentiment": model.classes_,
                        "Probability": probabilities
                    })

                    st.bar_chart(
                        probability_df.set_index("Sentiment")
                    )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee of correctness."
            )

# ---------------------------------
# Dataset Preview
# ---------------------------------
with st.expander("📁 View Dataset Preview"):
    st.dataframe(
        df[["review_text", "sentiment"]],
        use_container_width=True
    )

# ---------------------------------
# Footer
# ---------------------------------
st.divider()

st.caption(
    "NLP Sentiment Analysis | Python | NLTK | Scikit-learn | Streamlit"
)
