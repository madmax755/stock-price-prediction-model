import pandas as pd
import glob
from datetime import datetime

def load_stock_data(directory):
    all_data = {}
    for file in glob.glob(f"{directory}/*_data.csv"):
        ticker = file.split('/')[-1].split('_')[0]
        try:
            data = pd.read_csv(file, index_col='Date', parse_dates=True)
            all_data[ticker] = data
            print(f"Successfully loaded data for {ticker}")
        except Exception as e:
            print(f"Error loading data for {ticker}: {str(e)}")
    return all_data

def load_economic_data(file_path):
    try:
        data = pd.read_csv(file_path, index_col='date', parse_dates=True)
        print("Successfully loaded economic data")
        return data
    except Exception as e:
        print(f"Error loading economic data: {str(e)}")
        return None

def load_sentiment_data(file_path):
    try:
        # Read the file without specifying any index
        data = pd.read_csv(file_path)
        print("Successfully loaded sentiment data")
        print(f"Original sentiment data columns: {data.columns.tolist()}")
        
        # Check if 'publishedAt' column exists and rename it to 'date'
        if 'publishedAt' in data.columns:
            data = data.rename(columns={'publishedAt': 'date'})
            print("Renamed 'publishedAt' column to 'date'")
        elif 'date' not in data.columns:
            print("Warning: Neither 'date' nor 'publishedAt' column found in sentiment data.")
            print("First few rows of sentiment data:")
            print(data.head())
            return None

        # Convert 'date' column to datetime and set it as index
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        print(f"Final sentiment data columns: {data.columns.tolist()}")
        print("First few rows of processed sentiment data:")
        print(data.head())
        
        return data
    except Exception as e:
        print(f"Error loading sentiment data: {str(e)}")
        return None

def integrate_data(stock_data, economic_data, sentiment_data):
    integrated_data = {}
    for ticker, data in stock_data.items():
        print(f"Integrating data for {ticker}")
        
        # Ensure the index is datetime
        data.index = pd.to_datetime(data.index)
        
        # Merge stock data with economic indicators
        if economic_data is not None:
            merged = data.join(economic_data, how='left')
        else:
            merged = data
        
        # Merge with sentiment data
        if sentiment_data is not None:
            if ticker in sentiment_data.columns:
                merged = merged.join(sentiment_data[ticker].rename(f'{ticker}_sentiment'), how='left')
            else:
                print(f"Warning: No sentiment data found for {ticker}")
        
        integrated_data[ticker] = merged
        
    return integrated_data

def main():
    stock_data = load_stock_data('data')
    economic_data = load_economic_data('data/economic_indicators.csv')
    sentiment_data = load_sentiment_data('data/news_sentiment.csv')

    if sentiment_data is not None:
        print("\nSentiment Data Info:")
        print(sentiment_data.info())

    integrated_data = integrate_data(stock_data, economic_data, sentiment_data)

    for ticker, data in integrated_data.items():
        print(f"\nSaving integrated data for {ticker}")
        print(f"Columns: {data.columns.tolist()}")
        print(f"First few rows:\n{data.head()}")
        data.to_csv(f'data/{ticker}_integrated.csv')
        print(f"Integrated data for {ticker} saved successfully.")

if __name__ == "__main__":
    main()