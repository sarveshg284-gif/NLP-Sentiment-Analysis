
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load dataset
df = pd.read_csv("reviews.csv")

print("\nDataset Loaded Successfully!")
print(df.head())

print("\nDataset Information:")
print(df.shape)
print(df['sentiment'].value_counts())

# Preprocessing
stop_words = set(stopwords.words('english'))
stop_words -= {'not', 'no', 'nor'}

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = re.sub(r'<.*?>', '', str(text))
    text = text.lower()

    words = re.findall(r'\b[a-z]+\b', text)

    cleaned_words = []

    for word in words:
        if word not in stop_words:
            cleaned_words.append(
                lemmatizer.lemmatize(word)
            )

    return ' '.join(cleaned_words)


df['cleaned_review'] = df['review_text'].apply(
    preprocess_text
)

print("\nPreprocessing Completed!")
print(df[['review_text', 'cleaned_review']].head())

# Input and output
X = df['cleaned_review']
y = df['sentiment']

# Split dataset
# Small dataset: stratification is not possible
# because neutral has only one record.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF Conversion Completed!")
print("Feature Shape:", X_train_tfidf.shape)

# Naive Bayes
nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

nb_predictions = nb_model.predict(X_test_tfidf)

nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)

print("\nNaive Bayes Accuracy:",
      nb_accuracy * 100, "%")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_tfidf, y_train)

lr_predictions = lr_model.predict(X_test_tfidf)

lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)

print("\nLogistic Regression Accuracy:",
      lr_accuracy * 100, "%")

# Model comparison
results = pd.DataFrame({
    'Model': ['Naive Bayes', 'Logistic Regression'],
    'Accuracy': [
        nb_accuracy * 100,
        lr_accuracy * 100
    ]
})

print("\nModel Comparison:")
print(results)

# Accuracy graph
sns.barplot(
    data=results,
    x='Model',
    y='Accuracy',
    hue='Model',
    legend=False
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()

# Classification report
print("\nNaive Bayes Classification Report")

print(classification_report(
    y_test,
    nb_predictions,
    zero_division=0
))

print("\nLogistic Regression Classification Report")

print(classification_report(
    y_test,
    lr_predictions,
    zero_division=0
))

# Confusion matrix
labels = ['positive', 'negative', 'neutral']

cm = confusion_matrix(
    y_test,
    nb_predictions,
    labels=labels
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Live sentiment prediction
def predict_sentiment(text):

    cleaned_text = preprocess_text(text)

    vector = tfidf.transform([cleaned_text])

    nb_result = nb_model.predict(vector)[0]
    lr_result = lr_model.predict(vector)[0]

    print("\nYour Review:", text)

    print("Naive Bayes:", nb_result)
    print("Logistic Regression:", lr_result)


# User input
print("\n--- Sentiment Prediction ---")

while True:

    user_review = input(
        "\nEnter review (or type exit): "
    )

    if user_review.lower() == 'exit':
        print("Thank you!")
        break

    predict_sentiment(user_review)
