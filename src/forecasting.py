import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_time_series(df):
    """Aggregate historical MODIS observations monthly."""
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    monthly_data = df.groupby(pd.Grouper(key='acq_date', freq='ME')).agg({
        'frp': 'mean',
        'latitude': 'count', # Count of fires
        'brightness': 'mean',
        'high_risk': 'sum'
    }).reset_index()
    monthly_data.rename(columns={'latitude': 'fire_count'}, inplace=True)
    return monthly_data

def train_prophet_model(df):
    """Train Prophet forecasting model on historical monthly wildfire activity."""
    ts_data = prepare_time_series(df)
    
    # Prophet requires 'ds' (date) and 'y' (target)
    prophet_df = ts_data[['acq_date', 'fire_count']].rename(columns={'acq_date': 'ds', 'fire_count': 'y'})
    
    # Train model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    return model, ts_data

def generate_forecast(model, periods=6, freq='ME'):
    """Generate projected wildfire activity trends up to 6 months from today's date."""
    last_date = model.history['ds'].max()
    today = pd.Timestamp.today()
    target_date = today + pd.DateOffset(months=6)
    
    if target_date > last_date:
        diff_months = (target_date.year - last_date.year) * 12 + target_date.month - last_date.month
        periods = max(periods, diff_months)
        
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def evaluate_forecast(model, df):
    """Evaluate forecasting performance using MAE, RMSE."""
    ts_data = prepare_time_series(df)
    prophet_df = ts_data[['acq_date', 'fire_count']].rename(columns={'acq_date': 'ds', 'fire_count': 'y'})
    
    # Use in-sample prediction for metric evaluation
    forecast = model.predict(prophet_df)
    y_true = prophet_df['y']
    y_pred = forecast['yhat']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def engineer_forecasting_features(df):
    """Engineer lag-based temporal features and rolling averages."""
    ts_data = prepare_time_series(df)
    ts_data['rolling_avg_3m'] = ts_data['fire_count'].rolling(window=3).mean()
    ts_data['lag_1m'] = ts_data['fire_count'].shift(1)
    ts_data['lag_1y'] = ts_data['fire_count'].shift(12)
    
    # Seasonal indicator
    ts_data['month'] = ts_data['acq_date'].dt.month
    ts_data['is_peak_season'] = ts_data['month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    
    return ts_data
