import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def time_series_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split a time series DataFrame into train, validation, and test sets.
    
    Args:
    df (pandas.DataFrame): The input DataFrame, assumed to be sorted by date.
    train_ratio (float): Proportion of data for training set.
    val_ratio (float): Proportion of data for validation set.
    test_ratio (float): Proportion of data for test set.
    
    Returns:
    tuple: (train_df, val_df, test_df)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

# Usage example:
# Assuming 'df' is your preprocessed DataFrame with a DatetimeIndex
train_data, val_data, test_data = time_series_split(pd.read_csv('clean featured ticker data/TSLA_clean_data.csv'))

import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(X):
    plt.figure(figsize=(20, 16))
    corr = X.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.show()

    # Print highly correlated feature pairs
    print("Highly correlated feature pairs:")
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                print(f"{corr.columns[i]} - {corr.columns[j]}: {corr.iloc[i, j]:.2f}")


from sklearn.metrics import mean_squared_error, mean_absolute_error

def moving_average_model(data, target_column, window_sizes=[5, 20, 50]):
    """
    Implement a simple moving average prediction model.
    
    Args:
    data (pandas.DataFrame): The input data
    target_column (str): The name of the column to predict
    window_sizes (list): List of window sizes to use for moving averages
    
    Returns:
    dict: A dictionary of results for each window size
    """
    results = {}
    
    for window in window_sizes:
        # Calculate moving average
        data[f'MA_{window}'] = data[target_column].rolling(window=window).mean()
        
        # Shift to use past data to predict future
        data[f'MA_{window}_pred'] = data[f'MA_{window}'].shift(1)
        
        # Remove NaN values
        valid_data = data[[target_column, f'MA_{window}_pred']].dropna()
        
        # Calculate errors
        mse = mean_squared_error(valid_data[target_column], valid_data[f'MA_{window}_pred'])
        mae = mean_absolute_error(valid_data[target_column], valid_data[f'MA_{window}_pred'])
        
        results[window] = {'MSE': mse, 'MAE': mae}
    
    return results

# Usage
# Assuming 'train_data' is your training DataFrame and 'Close' is the target column
ma_results = moving_average_model(train_data, 'Close')
print(ma_results)


train_data, val_data, test_data = time_series_split(pd.read_csv('clean featured ticker data/TSLA_clean_data.csv'))


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_data_for_lr(df, target_column):
    """
    Prepare training and validation data for linear regression.
    
    Args:
    train_df (pandas.DataFrame): Training data
    val_df (pandas.DataFrame): Validation data
    target_column (str): Name of the target column
    
    Returns:
    tuple: (X_train, y_train, X_val, y_val, scaler)
    """
    
    df = df.drop(columns='TSLA_sentiment').dropna()

    

    # Separate features and target
    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    
    return X_train_scaled, y_train, scaler

# Usage
X_train, y_train, scaler = prepare_data_for_lr(train_data, 'Close')
X_val, y_val, scaler = prepare_data_for_lr(val_data, 'Close')


def linear_regression_model(X_train, y_train, X_val, y_val):
    """
    Implement a linear regression model.
    
    Args:
    X_train, X_val (pandas.DataFrame): Training and validation feature sets (already scaled)
    y_train, y_val (pandas.Series): Training and validation target variables
    
    Returns:
    tuple: (model, results_dict)
    """
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    val_mse = mean_squared_error(y_val, y_pred_val)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    
    results = {
        'Train MSE': train_mse,
        'Validation MSE': val_mse,
        'Train MAE': train_mae,
        'Validation MAE': val_mae,
        'Train R2': train_r2,
        'Validation R2': val_r2
    }
    
    return model, results

# Usage
lr_model, lr_results = linear_regression_model(X_train, y_train, X_val, y_val)
print(lr_results)

testX, testY, scaler = prepare_data_for_lr(test_data, 'Close')

def test_on_set(model, testX, testY):
    pred_y = model.predict(testX)

    mse = mean_squared_error(testY, pred_y)
    mae = mean_absolute_error(testY, pred_y)
    r2 = r2_score(testY, pred_y)
    
    results = {
        'Test MSE': mse,
        'Test MAE': mae,
        'Test R2': r2
    }
    
    return results, pred_y

results, pred_y = test_on_set(lr_model, testX, testY)


print(results)
# # Print feature importance
# feature_importance = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': lr_model.coef_
# }).sort_values('importance', ascending=False)
# print(feature_importance)

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

def train_regularized_model(X_train, y_train, X_val, y_val, model_type='ridge'):
    if model_type == 'ridge':
        model = Ridge()
        param_grid = {'alpha': [0.1, 1, 10, 100]}
    elif model_type == 'lasso':
        model = Lasso()
        param_grid = {'alpha': [0.1, 1, 10, 100]}
    else:
        raise ValueError("model_type must be 'ridge' or 'lasso'")

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_val = best_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    print(f"Best {model_type} alpha: {grid_search.best_params_['alpha']}")
    print(f"Validation MSE: {mse}")
    print(f"Validation R2: {r2}")

    return best_model

ridge_model = train_regularized_model(X_train, y_train, X_val, y_val, 'ridge')
lasso_model = train_regularized_model(X_train, y_train, X_val, y_val, 'lasso')


def check_data_consistency(train_df, val_df, test_df, target_column='Close'):
    print("Date ranges:")
    print(f"Train: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Validation: {val_df.index.min()} to {val_df.index.max()}")
    print(f"Test: {test_df.index.min()} to {test_df.index.max()}")
    
    print("\nFeature statistics:")
    for df, name in [(train_df, 'Train'), (val_df, 'Validation'), (test_df, 'Test')]:
        print(f"\n{name} set:")
        print(df[target_column].describe())
   
train_data, val_data, test_data = time_series_split(pd.read_csv('clean featured ticker data/TSLA_clean_data.csv'))
check_data_consistency(train_data, val_data, test_data)






