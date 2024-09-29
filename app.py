import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import warnings

warnings.filterwarnings('ignore')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Data Preprocessing
def no_of_words(text):
    words = text.split()
    return len(words)
# Data Preprocessing
def data_processing(text):
    import nltk
    nltk.download('punkt')  # Download the 'punkt' tokenizer
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


def stemming(data):
    return [stemmer.stem(word) for word in data]

# Streamlit Interface
st.title("IMDB Sentiment Analysis Web App")

# File Upload
uploaded_file = st.file_uploader("Upload your IMDB Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Data Information
    st.subheader("Data Overview")
    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment', data=df)
    plt.title("Sentiment distribution")
    st.pyplot(fig)

    # Data Cleaning and Preprocessing
    df['review'] = df['review'].apply(data_processing)
    df['word count'] = df['review'].apply(no_of_words)
    df = df.drop_duplicates('review')

    # Word Cloud for Positive and Negative Reviews
    pos_reviews = df[df.sentiment == 'positive']
    neg_reviews = df[df.sentiment == 'negative']

    if st.checkbox("Show WordCloud for Positive Reviews"):
        text = ' '.join([word for word in pos_reviews['review']])
        wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    if st.checkbox("Show WordCloud for Negative Reviews"):
        text = ' '.join([word for word in neg_reviews['review']])
        wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    # Common Words in Positive and Negative Reviews
    st.subheader("Common Words in Reviews")
    if st.checkbox("Show Common Words for Positive Reviews"):
        count = Counter()
        for text in pos_reviews['review'].values:
            for word in text.split():
                count[word] += 1
        pos_words = pd.DataFrame(count.most_common(15), columns=['word', 'count'])
        fig = px.bar(pos_words, x='count', y='word', title='Common words in positive reviews', color='word')
        st.plotly_chart(fig)

    if st.checkbox("Show Common Words for Negative Reviews"):
        count = Counter()
        for text in neg_reviews['review'].values:
            for word in text.split():
                count[word] += 1
        neg_words = pd.DataFrame(count.most_common(15), columns=['word', 'count'])
        fig = px.bar(neg_words, x='count', y='word', title='Common words in negative reviews', color='word')
        st.plotly_chart(fig)

    # Train-Test Split and Model Selection
    X = df['review']
    Y = df['sentiment'].replace({"positive": 1, "negative": 0})

    vect = TfidfVectorizer()
    X = vect.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    st.subheader("Model Training and Evaluation")

    # Logistic Regression
    if st.checkbox("Train Logistic Regression Model"):
        logreg = LogisticRegression()
        logreg.fit(x_train, y_train)
        logreg_pred = logreg.predict(x_test)
        logreg_acc = accuracy_score(logreg_pred, y_test)
        st.write(f"Logistic Regression Test Accuracy: {logreg_acc*100:.2f}%")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, logreg_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, logreg_pred))

    # Multinomial Naive Bayes
    if st.checkbox("Train Multinomial Naive Bayes Model"):
        mnb = MultinomialNB()
        mnb.fit(x_train, y_train)
        mnb_pred = mnb.predict(x_test)
        mnb_acc = accuracy_score(mnb_pred, y_test)
        st.write(f"Multinomial Naive Bayes Test Accuracy: {mnb_acc*100:.2f}%")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, mnb_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, mnb_pred))

    # Support Vector Classifier (SVC)
    if st.checkbox("Train Support Vector Classifier (SVC)"):
        svc = LinearSVC()
        svc.fit(x_train, y_train)
        svc_pred = svc.predict(x_test)
        svc_acc = accuracy_score(svc_pred, y_test)
        st.write(f"SVC Test Accuracy: {svc_acc*100:.2f}%")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, svc_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, svc_pred))
