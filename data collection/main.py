import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from typing import Tuple, Dict
from fetch_bond_data import fetch_treasury_yield_data
from fetch_stock_data import fetch_data_from_all_sources
from fetch_economic_data import fetch_economic_data

def fetch_stock_data_from_API(ticker: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    return fetch_data_from_all_sources(ticker, start_date, end_date)

def fetch_bond_data_from_API(start_date: str, end_date: str):
    return fetch_treasury_yield_data(start_date, end_date)

def fetch_economic_data_from_API(start_date: str, end_data: str):
    return fetch_economic_data(start_date, end_data)

def merge_data(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    # Ensure all dataframes have only a date as index
    for index, df in enumerate(dataframes):
        dataframes[index] = df.set_index(pd.to_datetime(df.index).date)
        dataframes[index].index.name = 'Date'

    # Merge all dataframes
    merged_data = pd.concat(dataframes, axis='columns')

    return merged_data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    # fill selected cols

    columns_to_fill = df.columns
    df = df[columns_to_fill].ffill()

    
    # to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Compute returns
    df_copy['Returns'] = df['Close'].pct_change()
    
    # df.dropna()

    return df_copy



def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Simple Moving Average
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Relative Strength Index
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
    
    return df


def select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].dropna()


# def split_data(df: pd.DataFrame, target_col: str = 'Returns') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     X = df.drop(columns=[target_col])
#     y = df[target_col]
#     return train_test_split(X, y, test_size=0.2, shuffle=False)


# def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
#     model = LinearRegression()
#     return model.fit(X_train, y_train)


# def evaluate_model(model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     return np.sqrt(mse)


# def generate_report(rmse: float) -> str:
#     return f"Model RMSE: {rmse}"

    
if __name__ == "__main__":
    start_date = "2020-01-01"
    end_date = "2023-06-01"
    use_stored_data = True
    ticker = "AAPL"

    if not use_stored_data:
        stock_data = fetch_stock_data_from_API(ticker, start_date, end_date)['yahoo']
        economic_data = fetch_economic_data_from_API(start_date, end_date)
        bond_data = fetch_bond_data_from_API(start_date, end_date)
        merged_data = merge_data([stock_data, economic_data, bond_data])
        data_with_indicators = compute_indicators(merged_data)
        preprocessed_data = preprocess_data(data_with_indicators)
    else:
        preprocessed_data = pd.read_csv(f'data/{ticker}_preprocessed_data.csv')
    
    final_features = select_features(preprocessed_data, ['Close', 'Volume', 'SMA_20', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'Returns'])
    # X_train, X_test, y_train, y_test = split_data(final_features)
    # model = train_model(X_train, y_train)
    # rmse = evaluate_model(model, X_test, y_test)
    # report = generate_report(rmse)