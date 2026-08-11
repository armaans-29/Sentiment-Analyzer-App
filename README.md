# Sentiment Analyzer
check it out here !!   https://goods-sentiment-analyzer.streamlit.app/

A Streamlit app that takes any CSV containing customer reviews or star ratings and automatically determines overall sentiment — no manual column selection needed. Built with Python, NLTK (VADER), and WordCloud.

## What it does

Upload a CSV and the app:

- Auto-detects whether the file has a text review column or a numeric rating column
- If text reviews are found, runs VADER sentiment analysis on each row to get positive/negative/neutral scores
- If only ratings are found (no review text), falls back to a rating-based sentiment mapping
- Shows the overall sentiment verdict and a score distribution chart
- Surfaces the top 3 most positive and most negative reviews
- Generates a word cloud from all review text to visualize common themes
- Lets you download the full dataset with sentiment scores appended as a new CSV

## How it works

**Column detection**
Column names are normalized (lowercased, spaces → underscores) and matched against common aliases:
- Review text: `review`, `reviews`, `review_text`, `comment`, `feedback`, `text`, `verified_reviews`
- Rating: `rating`, `ratings`, `stars`, `review_rating`, `product_rating`

If no header matches, the app falls back to heuristics — the longest average-length text column for reviews, or a numeric column bounded between 0 and 5 for ratings.

**Text-based sentiment**
Each review is scored with NLTK's VADER `SentimentIntensityAnalyzer`, which returns positive, negative, and neutral polarity scores per row. These are summed across the dataset to determine the overall sentiment label.

**Rating-based sentiment (fallback)**
When there's no review text, ratings are bucketed into sentiment classes:

| Rating | Sentiment |
|---|---|
| > 4.5 | Positive |
| 4.2 – 4.5 | Neutral |
| < 4.2 | Negative |

## Tech stack

- Python
- Streamlit
- Pandas
- NLTK (VADER Sentiment)
- WordCloud
- Matplotlib

## Running it locally

```bash
git clone https://github.com/<your-username>/csv-sentiment-analyzer.git
cd csv-sentiment-analyzer

pip install streamlit pandas nltk wordcloud matplotlib
streamlit run app.py
```

The app downloads the VADER lexicon automatically on first run. Then open `http://localhost:8501` and upload a CSV.

## Input format

No strict schema required. The app accepts:
- A CSV with a text review column (any of the aliases above, or the longest text column found), **or**
- A CSV with a numeric rating column (0–5 scale)

Both UTF-8 and Latin-1 encoded files are supported, with automatic fallback if UTF-8 decoding fails.

## Output

- Overall sentiment verdict (Positive / Neutral / Negative)
- Bar chart of aggregate sentiment scores
- Table of average sentiment score per entry
- Top 3 positive and negative reviews (text-based mode only)
- Word cloud of all review content (text-based mode only)
- Downloadable CSV with per-row sentiment scores added

## Limitations

- VADER is a lexicon-based sentiment tool tuned for short, informal text (like reviews and social posts) — it may misjudge sarcasm, domain-specific jargon, or very long-form text
- The rating-based fallback uses fixed thresholds (4.2 / 4.5) that were chosen as reasonable defaults, not derived from a specific dataset
- Word cloud and top review sections only appear when actual review text is available, not in ratings-only mode

## Project structure

```
├── app.py
├── requirements.txt
└── README.md
```
