import pandas as pd


tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
for ticker in tickers:

    # ----------------------  CLEANING DATA ------------------------------

    df = pd.read_csv(f'data/{ticker}_integrated.csv')


    # Identify columns to clean (excluding sentiment - can currently only get 30 days of data wihtout scraping myself)
    columns_to_clean = [col for col in df.columns if col != f"{ticker}_sentiment"]

    # Fill missing values with forward fill method
    df[columns_to_clean] = df[columns_to_clean].ffill()

    # If there are still missing values at the beginning, use backward fill
    df[columns_to_clean] = df[columns_to_clean].bfill()

    # unsure whether to remove outliers or not
    # pros vs cons - compare models trained on both??


    # -------------------- ENSURING CONSISTENCY OF TYPES ----------------------

    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date
    df = df.sort_values('Date')

    # Reset index
    df = df.reset_index(drop=True)

    # Ensure all numeric columns are of type float
    for col in ['Open','High','Low','Close','Volume','Dividends','Stock Splits','GDP','Unemployment_Rate','Federal_Funds_Rate',f'{ticker}_sentiment']:
        df[col] = df[col].astype(float)
    

    df.to_csv(f'clean ticker data/{ticker}_clean_data.csv')
