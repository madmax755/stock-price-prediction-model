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
    



    # --------------------- CALCUATE INDICATORS/FEATURES --------------------------------

    # Moving averages

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()


    # RSI

    def compute_rsi(data, time_window):
        diff = data.diff()
        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)
        
        avg_gain = gain.rolling(window=time_window).mean()
        avg_loss = loss.rolling(window=time_window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = compute_rsi(df['Close'], 14)



    # MACD

    def compute_macd(data, fast_period=12, slow_period=26, signal_period=9):
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    df['MACD'], df['Signal_Line'] = compute_macd(df['Close'])


    # Bollinger bands

    def compute_bollinger_bands(data, window=20, num_std=2):
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    df['BB_Upper'], df['BB_Lower'] = compute_bollinger_bands(df['Close'])



    # Momentum (pros:   a linear, more responsive momentum measure than RSI )
    df['Momentum'] = df['Close'] - df['Close'].shift(4)

    # Volatility (using Average True Range)
    df['HL'] = df['High'] - df['Low']
    df['HC'] = abs(df['High'] - df['Close'].shift(1))
    df['LC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()


    # ---------------------------- NORMALISING THE DATA -------------------------------
    # see future blog on normalisation (maxkendall.com)

   

    df.to_csv(f'clean featured ticker data/{ticker}_clean_data.csv')
