import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import os
import seaborn as sns
import pandas as pd

def get_color(risk):
    if risk == 'Low': return 'green'
    if risk == 'Medium': return 'orange'
    return 'red'

def create_fire_map(df, center=[22, 80], zoom=5, sample_size=5000):
    """Create a cleaner Folium HeatMap for fire hotspots."""
    fire_map = folium.Map(location=center, zoom_start=zoom, tiles='CartoDB dark_matter')
    
    sample_df = df.sample(min(len(df), sample_size))
    
    # Heatmap layer
    heat_data = [[row['latitude'], row['longitude'], row['frp']] for index, row in sample_df.iterrows()]
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(fire_map)
    
    # Add colored markers for highest risk (optional, just top 100 to avoid clutter)
    extreme_fires = sample_df[sample_df['high_risk'] == 1].nlargest(100, 'frp')
    for _, row in extreme_fires.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color='red',
            fill=True,
            fill_opacity=0.8,
            popup=f"Risk: High | FRP: {row['frp']}"
        ).add_to(fire_map)
        
    return fire_map

def plot_daily_trend(df):
    """Plot daily fire counts and 7-day moving average."""
    daily_counts = df.groupby('acq_date').size().reset_index(name='fire_count')
    daily_counts['moving_avg_7'] = daily_counts['fire_count'].rolling(7).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_counts['acq_date'], daily_counts['fire_count'], label='Actual Daily Count', alpha=0.4, color='#ff7f0e')
    ax.plot(daily_counts['acq_date'], daily_counts['moving_avg_7'], label='7-Day Moving Average', linewidth=2, color='#d62728')
    ax.set_title("Historical Daily Fire Trend", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Fire Count")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_forecast(forecast):
    """Plot Prophet forecast with confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Trend', color='#d62728', linewidth=2)
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='#ff9896', alpha=0.3, label='Confidence Interval')
    
    # Filter x-axis for readability (4 years historical + 6 months forecast)
    max_date = forecast['ds'].max()
    start_date = max_date - pd.DateOffset(years=4, months=6)
    ax.set_xlim([start_date, max_date])
    
    # Historical part? Usually prophet's plot method is easier but we do it custom for UI
    historical = forecast[forecast['trend'].notnull() & (forecast['ds'] < forecast['ds'].max() - pd.Timedelta(days=180))]
    
    ax.set_title("6-Month Wildfire Forecasting Projection", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Projected Fire Count (Monthly)")
    ax.legend()
    fig.tight_layout()
    return fig
