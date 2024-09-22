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
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'])
    return df.set_index('date')

def main():
    api_key = f"{os.getenv('FRED_API_KEY')}"
    indicators = {
        "GDP": "GDP",
        "Unemployment Rate": "UNRATE",
        "Federal Funds Rate": "FEDFUNDS"
    }
    start_date = "2018-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    economic_data = {}
    for name, code in indicators.items():
        economic_data[name] = fetch_economic_data(code, start_date, end_date, api_key)
        print(f"Data for {name} fetched successfully.")

    # Combine all indicators into one DataFrame
    combined_data = pd.concat(economic_data.values(), axis=1, keys=economic_data.keys())
    combined_data.to_csv("data/economic_indicators.csv")
    print("Economic indicators data saved successfully.")

if __name__ == "__main__":
    main()