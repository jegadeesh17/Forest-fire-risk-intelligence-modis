import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def get_season(month):
    if month in [12, 1, 2]:
        return 1  # winter
    elif month in [3, 4, 5]:
        return 2  # spring
    elif month in [6, 7, 8]:
        return 3  # summer
    else:
        return 4  # autumn

def create_features(df):
    """Generate temporal, spatial, and risk-based features."""
    # 1. Temporal features
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    df['year'] = df['acq_date'].dt.year
    df['month'] = df['acq_date'].dt.month
    df['day'] = df['acq_date'].dt.day
    
    # 2. Time features
    df['acq_time_str'] = df['acq_time'].astype(int).astype(str).str.zfill(4)
    df['hours'] = df['acq_time_str'].str[:2].astype(int)
    df['minutes'] = df['acq_time_str'].str[2:].astype(int)
    
    # 3. Season
    df['season'] = df['month'].apply(get_season)
    
    # 4. Spatial Clustering
    kmeans = KMeans(n_clusters=15, random_state=42)
    df['region_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
    
    # Map clusters to readable pseudo-states based on centroids
    cluster_names = {
        0: "Punjab & Haryana", 1: "Odisha & Chhattisgarh", 2: "Mizoram & Tripura",
        3: "Western Maharashtra", 4: "Western Madhya Pradesh", 5: "Eastern Madhya Pradesh",
        6: "Tamil Nadu", 7: "UP & Nepal Border", 8: "Meghalaya & Assam",
        9: "Marathwada & Telangana", 10: "Gujarat", 11: "Nagaland & Upper Assam",
        12: "Jharkhand", 13: "Bastar & Eastern Telangana", 14: "Rayalaseema (AP)"
    }
    df['region_name'] = df['region_cluster'].map(cluster_names)
    
    # 5. Fire Activity counts
    df['cluster_fire_count'] = df.groupby('region_cluster')['region_cluster'].transform('count')
    df['monthly_cluster_fire_count'] = df.groupby(['region_cluster', 'month'])['region_cluster'].transform('count')
    
    # 6. Thermal Anomalies
    df['brightness_diff'] = df['brightness'] - df['bright_t31']
    
    # 7. Targets and Categories
    df['high_risk'] = ((df['confidence'] > 80) & (df['frp'] > 20)).astype(int)
    
    df['frp_category'] = pd.cut(
        df['frp'],
        bins=[0, 10, 30, 60, 1000],
        labels=['Low', 'Medium', 'High', 'Extreme']
    )
    
    # Sort and reset index
    df = df.sort_values(['year', 'month', 'day', 'hours', 'minutes']).reset_index(drop=True)
    
    return df
