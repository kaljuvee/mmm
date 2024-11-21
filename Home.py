import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Marketing Mix Model Analysis")

st.title("Marketing Mix Model (MMM) Analysis Tool")

# Sidebar for data upload and configuration
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload your marketing data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        
        # Data configuration
        target_col = st.selectbox("Select target variable (e.g., ROI, Sales)", df.columns)
        media_channels = st.multiselect("Select media channels", 
                                      [col for col in df.columns if col != target_col],
                                      default=[col for col in df.columns if col != target_col][:3])

# Main content
if 'df' in locals():
    # Data Overview
    st.header("Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df.head())
    with col2:
        st.write("Summary Statistics")
        st.dataframe(df.describe())

    # Correlation Analysis
    st.header("Correlation Analysis")
    corr_matrix = df[media_channels + [target_col]].corr()
    
    # Heatmap
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns)
    st.plotly_chart(fig, use_container_width=True)

    # Model Training
    st.header("Marketing Mix Model")
    
    X = df[media_channels]
    y = df[target_col]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Channel': media_channels,
        'Coefficient': model.coef_
    })
    importance_df = importance_df.sort_values('Coefficient', ascending=True)
    
    # Plot feature importance
    fig = px.bar(importance_df, x='Coefficient', y='Channel', orientation='h',
                 title='Channel Impact on ' + target_col)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive What-If Analysis
    st.header("What-If Analysis")
    st.write("Adjust channel spending to see the impact on " + target_col)
    
    col1, col2 = st.columns(2)
    
    # Current values
    current_values = {}
    adjusted_values = {}
    
    with col1:
        st.subheader("Current Values")
        for channel in media_channels:
            current_values[channel] = df[channel].mean()
            st.metric(channel, f"{current_values[channel]:.2f}")
    
    with col2:
        st.subheader("Adjust Values")
        for channel in media_channels:
            adjusted_values[channel] = st.slider(
                f"Adjust {channel}",
                min_value=float(df[channel].min()),
                max_value=float(df[channel].max()),
                value=float(df[channel].mean())
            )
    
    # Calculate predictions
    current_pred = model.predict(scaler.transform(pd.DataFrame([current_values])))[0]
    adjusted_pred = model.predict(scaler.transform(pd.DataFrame([adjusted_values])))[0]
    
    # Display impact
    st.header("Impact Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current Predicted " + target_col, f"{current_pred:.2f}")
    with col2:
        st.metric("Adjusted Predicted " + target_col, f"{adjusted_pred:.2f}", 
                 delta=f"{adjusted_pred - current_pred:.2f}")

    # Time Series Analysis (if date column exists)
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        st.header("Time Series Analysis")
        date_col = date_cols[0]
        
        fig = px.line(df, x=date_col, y=[target_col] + media_channels,
                     title=f"Time Series View of {target_col} and Channel Spending")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload your marketing data CSV file to begin analysis.")
    
    # Sample data format
    st.header("Expected Data Format")
    sample_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=5),
        'TV_Spend': [1000, 1200, 800, 1500, 1300],
        'Radio_Spend': [500, 600, 400, 700, 550],
        'Social_Media_Spend': [800, 900, 750, 1000, 850],
        'ROI': [1.5, 1.8, 1.2, 2.0, 1.7]
    })
    st.dataframe(sample_data)
