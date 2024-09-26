import pandas as pd
import numpy as np

aapl_df = pd.read_csv('clean featured ticker data/AAPL_clean_data.csv')
googl_df = pd.read_csv('clean featured ticker data/GOOGL_clean_data.csv')
msft_df = pd.read_csv('clean featured ticker data/MSFT_clean_data.csv')
tsla_df = pd.read_csv('clean featured ticker data/TSLA_clean_data.csv')


print(aapl_df.describe())

correlation_matrix = aapl_df.corr()

high_correlations = correlation_matrix[abs(correlation_matrix) > 0.7].stack()
high_correlations = high_correlations[high_correlations < 1].sort_values(ascending=False)
print("Highly correlated features:\n", high_correlations)

daily_log_returns = aapl_df['Close']/aapl_df['Close'].shift(1) - 1
daily_log_returns = daily_log_returns.apply(lambda x: np.log(1+x))

print(daily_log_returns.sum()/aapl_df['Close'].count()*365)


# rsi investigation
import matplotlib.pyplot as plt
plt.plot(aapl_df.index, aapl_df['RSI'])
plt.plot(aapl_df.index, (np.exp(daily_log_returns)-1)*1000+50)
plt.plot(aapl_df.index, [50 for _ in aapl_df['Close'].values ])
plt.show()

corr_by_shift = [pd.DataFrame(aapl_df['RSI']).join(aapl_df['Close'].shift(-x)).corr().iloc[0]['Close'] for x in range(-10, 10)]
print(corr_by_shift)