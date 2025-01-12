from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import PyPDF2
import pandas as pd
from io import StringIO
import ssl
import nltk

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize Flask and NLTK
app = Flask(__name__)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Sentiment Analysis Function
def analyze_sentiment(text):
    sentences = nltk.sent_tokenize(text)
    results = []
    for sentence in sentences:
        sentiment_score = sia.polarity_scores(sentence)
        results.append({
            'text': sentence,
            'positive': sentiment_score['pos'],
            'neutral': sentiment_score['neu'],
            'negative': sentiment_score['neg'],
            'compound': sentiment_score['compound']
        })
    return pd.DataFrame(results)

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename.endswith('.pdf'):
            raw_text = extract_text_from_pdf(uploaded_file)
        else:
            raw_text = uploaded_file.read().decode("utf-8")

        # Process and analyze text
        preprocessed_text = preprocess_text(raw_text)
        sentiment_df = analyze_sentiment(preprocessed_text)

        # Render results
        return render_template('results.html', tables=[sentiment_df.to_html(classes='data', header=True)])

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
