import pandas as pd

def generate_insights(df, forecast):
    """Generate dynamic AI risk insights based on historical and forecasted data."""
    insights = []
    
    # 1. Historical Peak Season
    monthly_fires = df.groupby(df['acq_date'].dt.month).size()
    peak_month = monthly_fires.idxmax()
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    
    insights.append(f"🔥 Historical data indicates that **{month_names[peak_month]}** is the peak month for wildfire activity.")
    
    # 2. Forecast Trend Analysis
    last_hist_date = df['acq_date'].max()
    future_forecast = forecast[forecast['ds'] > last_hist_date]
    
    if not future_forecast.empty:
        trend_diff = future_forecast['yhat'].iloc[-1] - future_forecast['yhat'].iloc[0]
        if trend_diff > 0:
            insights.append(f"📈 Forecast models predict an **escalating trend** (+{trend_diff:.0f} fires/month) in wildfire activity over the next 6 months.")
        else:
            insights.append(f"📉 Forecast models project a **decrease** in wildfire events for the upcoming 6-month period.")
            
        peak_forecast_month = future_forecast.loc[future_forecast['yhat'].idxmax()]
        insights.append(f"⚠️ Extreme caution advised for **{peak_forecast_month['ds'].strftime('%B %Y')}**, which shows the highest projected risk ({peak_forecast_month['yhat']:.0f} anticipated events).")
    
    # 3. High Risk Regions
    if 'region_cluster' in df.columns:
        top_region = df[df['high_risk'] == 1]['region_cluster'].mode()[0]
        insights.append(f"📍 **Region {top_region}** has historically concentrated the most high-severity thermal anomalies (FRP > 20, Confidence > 80%).")
        
    return insights
