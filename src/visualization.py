import matplotlib.pyplot as plt
import folium
import os

def create_fire_map(df, center=[22, 80], zoom=5, sample_size=5000):
    """Create a Folium map showing fire hotspots."""
    fire_map = folium.Map(location=center, zoom_start=zoom)
    
    sample_df = df.sample(min(len(df), sample_size))
    for _, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.6,
            popup=f"FRP: {row['frp']}"
        ).add_to(fire_map)
    return fire_map

def plot_daily_trend(df, output_path=None):
    """Plot daily fire counts and 7-day moving average."""
    daily_counts = df.groupby('acq_date').size().reset_index(name='fire_count')
    daily_counts['moving_avg_7'] = daily_counts['fire_count'].rolling(7).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_counts['acq_date'], daily_counts['fire_count'], label='Actual Daily Count', alpha=0.5)
    plt.plot(daily_counts['acq_date'], daily_counts['moving_avg_7'], label='7-Day Moving Average', linewidth=2)
    plt.title("Daily Fire Trend Analysis")
    plt.xlabel("Date")
    plt.ylabel("Fire Count")
    plt.legend()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
