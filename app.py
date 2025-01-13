import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

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

# Load the CSV containing hardcoded reasoning
reasoning_csv_path = "hardcoded_reasoning.csv"  # Update with your actual CSV file path
reasoning_df = pd.read_csv(reasoning_csv_path)


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

        # Detect Campaign Name
        elif "Campaign Name:" in line:
            campaign_match = re.search(r'Campaign Name:\s*["“](.+?)["”]', line)
            if campaign_match:
                campaign_name = campaign_match.group(1)

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

        # Collect Multiline Slogan
        elif collecting_slogan:
            if line.endswith('"') or line.endswith('”'):  # End of multiline slogan
                slogan_text += " " + line.rstrip('"').rstrip('”')
                collecting_slogan = False
            else:
                slogan_text += " " + line  # Continue collecting slogan

        # Finalize Entry
        if not collecting_slogan and slogan_text:
            # Perform Sentiment Analysis
            sentiment_score = analyzer.polarity_scores(slogan_text)
            sentiment_category = categorize_sentiment(sentiment_score['compound'])

            # Fetch Hardcoded Reasoning
            reasoning = fetch_reasoning(current_brand, campaign_name, slogan_text)

            # Append Entry
            entries.append({
                'Brand Name': current_brand,
                'Campaign Name': campaign_name,
                'Slogan/Textual Content': slogan_text,
                'Compound Score': sentiment_score['compound'],
                'Sentiment Category': sentiment_category,
                'Reasoning': reasoning
            })

            # Reset slogan_text after adding the entry
            slogan_text = ""

    return pd.DataFrame(entries)


def fetch_reasoning(brand_name, campaign_name, slogan_text):
    """
    Fetch the hardcoded reasoning from the CSV file based on the brand name, campaign name, or slogan.
    """
    filtered_df = reasoning_df[
        (reasoning_df['Brand Name'] == brand_name) &
        (reasoning_df['Campaign Name'] == campaign_name)
    ]

    # Check if the column exists and has data
    if 'AI' in filtered_df.columns and not filtered_df.empty:
        return filtered_df['AI'].iloc[0]
    else:
        return "No reasoning available."



def categorize_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


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

print(reasoning_df.head())
print(reasoning_df.columns)

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
