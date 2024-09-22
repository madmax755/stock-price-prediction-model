import pandas as pd
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_economic_data(indicator, start_date, end_date, api_key):
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": indicator,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    df = pd.DataFrame(data['observations'])

    # convert to datetime and removee timesone informations
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.set_index('date')['value']

def main():
    api_key = f"{os.getenv('FRED_API_KEY')}"
    indicators = {
        "GDP": "GDP",
        "Unemployment_Rate": "UNRATE",
        "Federal_Funds_Rate": "FEDFUNDS"
    }
    start_date = "2018-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    economic_data = pd.DataFrame()
    for name, code in indicators.items():
        series = fetch_economic_data(code, start_date, end_date, api_key)
        economic_data[name] = series
        print(f"Data for {name} fetched successfully.")

    # Ensure the index name is 'date'
    economic_data.index.name = 'date'

    # Sort the data by date
    economic_data.sort_index(inplace=True)

    # Save the data
    os.makedirs('data', exist_ok=True)
    economic_data.to_csv("data/economic_indicators.csv")
    print("Economic indicators data saved successfully.")
    print(economic_data.head())

if __name__ == "__main__":
    main()