import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import openai
from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
import pdfplumber
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re
from io import BytesIO
import ssl

# SSL Fix for NLTK Downloads (macOS)
ssl._create_default_https_context = ssl._create_unverified_context

# Download required NLTK resources
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

# Set OpenAI API Key (Replace with your key)
client = openai.Client(api_key="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")


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            text += page_text + "\n"
    return text


def process_pdf_text(text):
    entries = []
    lines = text.splitlines()
    current_brand = ""
    campaign_name = ""
    slogan_text = ""
    collecting_slogan = False  # Flag to handle multiline slogans

    for line in lines:
        line = line.strip()

        # Detect Brand Name
        if re.match(r'^\d+\.', line):
            current_brand = line
            collecting_slogan = False  # Reset flag
            slogan_text = ""  # Reset slogan
            print(f"Brand Detected: {current_brand}")

        # Detect Campaign Name
        elif "Campaign Name:" in line:
            campaign_match = re.search(r'Campaign Name:\s*["“](.+?)["”]', line)
            if campaign_match:
                campaign_name = campaign_match.group(1)
            print(f"Campaign Name Detected: {campaign_name}")

        # Detect Slogan/Textual Content
        elif "Slogan/Textual Content:" in line:
            slogan_match = re.search(r'Slogan/Textual Content:\s*["“](.+)', line)
            if slogan_match:
                slogan_text = slogan_match.group(1)  # Start collecting slogan
                if line.endswith('"') or line.endswith('”'):  # Single-line slogan ends here
                    slogan_text = slogan_text.rstrip('"').rstrip('”')  # Remove closing quotes
                    collecting_slogan = False
                else:  # Multiline slogan starts
                    collecting_slogan = True
            print(f"Slogan Start Detected: {slogan_text}")

        # Collect Multiline Slogan
        elif collecting_slogan:
            if line.endswith('"') or line.endswith('”'):  # End of multiline slogan
                slogan_text += " " + line.rstrip('"').rstrip('”')
                collecting_slogan = False
            else:
                slogan_text += " " + line  # Continue collecting slogan

        # Finalize Entry
        if not collecting_slogan and slogan_text:
            print(f"Finalizing Entry for: {current_brand}")

            # Perform Sentiment Analysis
            sentiment_score = analyzer.polarity_scores(slogan_text)
            sentiment_category = categorize_sentiment(sentiment_score['compound'])

            # Generate AI Reasoning
            reasoning = get_ai_reasoning(slogan_text, sentiment_score['compound'])

            # Append Entry
            entries.append({
                'Brand Name': current_brand,
                'Campaign Name': campaign_name,
                'Slogan/Textual Content': slogan_text,
                'Compound Score': sentiment_score['compound'],
                'Sentiment Category': sentiment_category,
                'AI Reasoning': reasoning
            })
            print(f"Entry Added: {entries[-1]}")

            # Reset slogan_text after adding the entry
            slogan_text = ""

    print(f"Total Entries Processed: {len(entries)}")
    return pd.DataFrame(entries)




def categorize_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def get_ai_reasoning(slogan, sentiment_score):
    sentiment_label = "positive" if sentiment_score > 0.05 else "negative" if sentiment_score < -0.05 else "neutral"

    messages = [
        {"role": "system", "content": "You are a marketing analyst assistant specializing in creating concise and impactful explanations of advertisement effectiveness."},
        {"role": "user", "content": f"The slogan '{slogan}' received a {sentiment_label} sentiment score of {sentiment_score}. Provide a short, attractive, and compelling explanation of why this slogan evokes a {sentiment_label} sentiment from customers in under 50 words."}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=50  # Limit to ensure brevity
    )

    return response.choices[0].message.content.strip()



def create_plot(df):
    if df.empty or 'Sentiment Category' not in df.columns:
        print("No data available for visualization.")
        return

    plt.figure(figsize=(8, 6))
    sentiment_counts = df['Sentiment Category'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Number of Campaigns')
    plt.xticks(rotation=0)
    plt.savefig('static/sentiment_plot.png')
    plt.close()


@app.route('/', methods=['GET', 'POST'])
def index():
    global latest_df

    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename.endswith('.pdf'):
            raw_text = extract_text_from_pdf(uploaded_file)
            sentiment_df = process_pdf_text(raw_text)
        elif uploaded_file.filename.endswith('.txt'):
            raw_text = uploaded_file.read().decode("utf-8")
            sentiment_df = process_pdf_text(raw_text)
        else:
            return "Unsupported file type. Please upload a .pdf or .txt file.", 400

        latest_df = sentiment_df
        print(f"DataFrame shape: {latest_df.shape}")  # Debugging

        create_plot(sentiment_df)
        return render_template('results.html', tables=[sentiment_df.to_html(classes='data', header=True)])

    return render_template('index.html')



@app.route('/download_csv')
def download_csv():
    output = BytesIO()
    latest_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='sentiment_results.csv')


if __name__ == "__main__":
    app.run(debug=True)