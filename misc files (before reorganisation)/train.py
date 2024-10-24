import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def time_series_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

def plot_correlation_matrix(X):
    plt.figure(figsize=(20, 16))
    corr = X.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.show()

    print("Highly correlated feature pairs:")
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                print(f"{corr.columns[i]} - {corr.columns[j]}: {corr.iloc[i, j]:.2f}")

def moving_average_model(data, target_column, window_sizes=[5, 20, 50]):
    results = {}
    
    for window in window_sizes:
        data[f'MA_{window}'] = data[target_column].rolling(window=window).mean()
        data[f'MA_{window}_pred'] = data[f'MA_{window}'].shift(1)
        valid_data = data[[target_column, f'MA_{window}_pred']].dropna()
        
        mse = mean_squared_error(valid_data[target_column], valid_data[f'MA_{window}_pred'])
        mae = mean_absolute_error(valid_data[target_column], valid_data[f'MA_{window}_pred'])
        
        results[window] = {'MSE': mse, 'MAE': mae}
    
    return results

def prepare_data_for_lr(df, target_column):
    df = df.drop(columns='TSLA_sentiment').dropna()
    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    
    return X_train_scaled, y_train, scaler

def linear_regression_model(X_train, y_train, X_val, y_val):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    results = {
        'Train MSE': mean_squared_error(y_train, y_pred_train),
        'Validation MSE': mean_squared_error(y_val, y_pred_val),
        'Train MAE': mean_absolute_error(y_train, y_pred_train),
        'Validation MAE': mean_absolute_error(y_val, y_pred_val),
        'Train R2': r2_score(y_train, y_pred_train),
        'Validation R2': r2_score(y_val, y_pred_val)
    }
    
    return model, results

def test_on_set(model, testX, testY):
    pred_y = model.predict(testX)
    
    results = {
        'Test MSE': mean_squared_error(testY, pred_y),
        'Test MAE': mean_absolute_error(testY, pred_y),
        'Test R2': r2_score(testY, pred_y)
    }
    
    return results, pred_y

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
