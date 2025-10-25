import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download VADER lexicon
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()

# Function to detect review column
def get_review_column(data):
    normalized = {col.strip().lower().replace(" ", "_"): col for col in data.columns}
    possible_reviews = ["verified_reviews", "review", "reviews", "review_text", "comment", "feedback", "text"]
    for pr in possible_reviews:
        if pr in normalized:
            return normalized[pr]
    text_cols = [col for col in data.columns if data[col].dtype == "object" and data[col].str.len().mean() > 3]
    return text_cols[0] if text_cols else None

# Function to detect rating column
def get_rating_column(data):
    normalized = {col.strip().lower().replace(" ", "_"): col for col in data.columns}
    possible_ratings = ["rating", "ratings", "stars", "review_rating", "product_rating"]
    for pr in possible_ratings:
        if pr in normalized:
            return normalized[pr]
    num_cols = data.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if data[col].max() <= 5 and data[col].min() >= 0:
            return col
    return None

# Analyze sentiment using text reviews
def analyze_text_sentiment(data, review_col):
    data[review_col] = data[review_col].astype(str)
    data["Positive"] = data[review_col].apply(lambda x: sentiments.polarity_scores(x)["pos"])
    data["Negative"] = data[review_col].apply(lambda x: sentiments.polarity_scores(x)["neg"])
    data["Neutral"] = data[review_col].apply(lambda x: sentiments.polarity_scores(x)["neu"])
    pos_sum = data["Positive"].sum()
    neg_sum = data["Negative"].sum()
    neu_sum = data["Neutral"].sum()
    overall = max({"Positive": pos_sum, "Neutral": neu_sum, "Negative": neg_sum}, key={"Positive": pos_sum, "Neutral": neu_sum, "Negative": neg_sum}.get)
    return overall, {"Positive": pos_sum, "Neutral": neu_sum, "Negative": neg_sum}, data

# Analyze sentiment using numeric ratings
def analyze_rating_sentiment(data, rating_col):
    def map_rating(r):
        if r > 4.5:
            return "Positive"
        elif r >= 4.2:
            return "Neutral"
        else:
            return "Negative"
    data["Rating Sentiment"] = data[rating_col].apply(map_rating)
    counts = data["Rating Sentiment"].value_counts().to_dict()
    counts = {k: counts.get(k, 0) for k in ["Positive", "Neutral", "Negative"]}
    total = sum(counts.values())
    overall = max(counts, key=counts.get)
    data["Positive"] = (data[rating_col] > 4.5).astype(int)
    data["Neutral"] = ((data[rating_col] >= 4.2) & (data[rating_col] <= 4.5)).astype(int)
    data["Negative"] = (data[rating_col] < 4.2).astype(int)
    return overall, counts, data

# Streamlit Page Configuration
st.set_page_config(page_title="📊 Sentiment Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .main { background-color: #FAF9F6; }
    h1 { color: #3b3b98; text-align:center; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; height: 3em; width: 15em; }
    .st-success { background-color: #e0ffe0 !important; border-left: 5px solid #4CAF50; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Sentiment Analyzer")
st.write("Upload any CSV containing reviews or ratings, and this app will automatically determine sentiment accordingly.")

uploaded_file = st.file_uploader("📂 Upload CSV (UTF-8 or Latin1 supported)", type=["csv"])

if uploaded_file:
    try:
        # Load with encoding fallback
        try:
            data = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            data = pd.read_csv(uploaded_file, encoding='latin1')

        review_col = get_review_column(data)
        rating_col = get_rating_column(data)

        if review_col:
            st.info(f"Detected review column: **{review_col}**")
            overall_sentiment, sentiment_scores, analyzed_data = analyze_text_sentiment(data, review_col)
        elif rating_col:
            st.info(f"Detected rating column: **{rating_col}**")
            overall_sentiment, sentiment_scores, analyzed_data = analyze_rating_sentiment(data, rating_col)
        else:
            st.error("❌ The dataset doesn't contain any appropriate review or rating columns.")
            st.stop()

        # Results Visualization
        st.success(f"Overall Sentiment: **{overall_sentiment}** ✅")
        st.subheader("📈 Sentiment Score Distribution")
        st.bar_chart(sentiment_scores)

        avg_scores = analyzed_data[["Positive", "Neutral", "Negative"]].mean()
        st.subheader("📊 Average Sentiment Scores per Entry")
        st.table(avg_scores.to_frame().rename(columns={0: "Average Score"}))

        if review_col:
            st.subheader("🌟 Top Positive Reviews")
            for rev in analyzed_data.sort_values("Positive", ascending=False)[review_col].head(3):
                st.write(f"✅ {rev}")

            st.subheader("💢 Top Negative Reviews")
            for rev in analyzed_data.sort_values("Negative", ascending=False)[review_col].head(3):
                st.write(f"❌ {rev}")

            # Word Cloud (if review content present)
            st.subheader("☁️ Word Cloud of Reviews")
            text = " ".join(analyzed_data[review_col].tolist())
            if text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                fig, ax = plt.subplots(figsize=(12,6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

        # Download section
        csv_data = analyzed_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Analyzed CSV", csv_data, "analyzed_sentiments.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error reading or analyzing CSV: {e}")