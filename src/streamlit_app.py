import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import os
import sys

# Ensure src directory is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocessing import preprocess_data
from feature_engineering import create_features
from visualization import create_fire_map, plot_daily_trend

st.set_page_config(page_title="Forest Fire Intelligence", page_icon="🔥", layout="wide")

st.title("🔥 Forest Fire Risk Intelligence")
st.write("Geospatial Analytics & Risk Dashboard")

@st.cache_data
def load_data():
    # Load sample dataset
    data_path = os.path.join(os.path.dirname(__file__), "../data/modis_data10%.csv")
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    df = preprocess_data(df)
    df = create_features(df)
    return df

with st.spinner("Loading Satellite Data..."):
    df = load_data()

if not df.empty:
    st.sidebar.header("Filter Options")
    selected_year = st.sidebar.selectbox("Select Year", df['year'].unique())
    selected_month = st.sidebar.selectbox("Select Month", df[df['year'] == selected_year]['month'].unique())
    
    filtered_df = df[(df['year'] == selected_year) & (df['month'] == selected_month)]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Hotspots (Selected)", len(filtered_df))
    col2.metric("High Risk Hotspots", len(filtered_df[filtered_df['high_risk'] == 1]))
    col3.metric("Avg FRP (Fire Radiative Power)", f"{filtered_df['frp'].mean():.2f}")
    
    st.subheader("🗺️ Geospatial Fire Distribution")
    st.write("Mapping thermal anomalies based on satellite telemetry.")
    # Create and display Folium Map
    fire_map = create_fire_map(filtered_df, center=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()], zoom=5, sample_size=1000)
    st_folium(fire_map, width=1200, height=500)
    
    st.subheader("📈 Temporal Fire Trends")
    # Using matplotlib as existing in src
    daily_counts = filtered_df.groupby('day').size().reset_index(name='fire_count')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_counts['day'], daily_counts['fire_count'], marker='o', color='firebrick')
    ax.set_title(f"Fire Occurrences - {selected_year}/{selected_month}")
    ax.set_xlabel("Day of Month")
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.info("No data available to display.")
