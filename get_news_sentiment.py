import pandas as pd
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_news(api_key, tickers, from_date, to_date):
    base_url = "https://newsapi.org/v2/everything"
    all_articles = []

    for ticker in tickers:
        params = {
            "q": ticker,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "apiKey": api_key
        }
        response = requests.get(base_url, params=params)
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error fetching news for {ticker}: Status code {response.status_code}")
            print(f"Response content: {response.text}")
            continue

        data = response.json()
        
        # Check if 'articles' key exists in the response
        if 'articles' not in data:
            print(f"No 'articles' found in response for {ticker}. Response data:")
            print(data)
            continue

        articles = data['articles']
        for article in articles:
            article['ticker'] = ticker
        all_articles.extend(articles)

    if not all_articles:
        raise ValueError("No articles were fetched. Please check your API key and request parameters.")

    return pd.DataFrame(all_articles)

def calculate_sentiment(text):
    if not isinstance(text, str):
        return 0  # Return neutral sentiment for non-string inputs
    return TextBlob(text).sentiment.polarity

def main():
    api_key = f"{os.getenv('NEWSAPI_API_KEY')}"
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    news_df = fetch_news(api_key, tickers, from_date, to_date)
    
    # Apply sentiment analysis, handling potential None values
    news_df['sentiment'] = news_df['description'].apply(calculate_sentiment)

    # Aggregate sentiment by date and ticker
    daily_sentiment = news_df.groupby(['ticker', pd.to_datetime(news_df['publishedAt']).dt.date])['sentiment'].mean().unstack(level=0)
    daily_sentiment.to_csv("data/news_sentiment.csv")
    print("News sentiment data saved successfully.")

if __name__ == "__main__":
    main()