import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from faker import Faker
import random

st.set_page_config(layout="wide", page_title="Marketing Mix Model Analysis")

# Function to generate synthetic data
def generate_synthetic_data(n_points=1000):
    fake = Faker()
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(n_points)]
    
    # Generate spending data with seasonal patterns and random variations
    data = {
        'Date': dates,
        'TV_Spend': [
            random.normalvariate(12000, 2000) * (1 + 0.3 * np.sin(2 * np.pi * i / 365))  # Annual seasonality
            for i in range(n_points)
        ],
        'Radio_Spend': [
            random.normalvariate(6000, 1000) * (1 + 0.2 * np.sin(2 * np.pi * i / 365))
            for i in range(n_points)
        ],
        'Social_Media_Spend': [
            random.normalvariate(9000, 1500) * (1 + 0.15 * np.sin(2 * np.pi * i / 90))  # Quarterly seasonality
            for i in range(n_points)
        ],
        'Digital_Spend': [
            random.normalvariate(7500, 1200) * (1 + 0.25 * np.sin(2 * np.pi * i / 90))
            for i in range(n_points)
        ]
    }
    
    # Generate sales with relationship to spending + noise
    base_sales = 25000
    data['Sales'] = [
        base_sales +
        0.8 * data['TV_Spend'][i] +
        1.2 * data['Radio_Spend'][i] +
        1.5 * data['Social_Media_Spend'][i] +
        1.3 * data['Digital_Spend'][i] +
        random.normalvariate(0, 5000)  # Add noise
        for i in range(n_points)
    ]
    
    return pd.DataFrame(data)

st.title("Marketing Mix Model (MMM) Analysis Tool")

# Sidebar for data upload and configuration
with st.sidebar:
    st.header("Data Configuration")
    
    # Radio button for data source selection
    data_source = st.radio(
        "Choose data source",
        ["Use sample data", "Generate synthetic data", "Upload your own data"]
    )
    
    if data_source == "Upload your own data":
        uploaded_file = st.file_uploader("Upload your marketing data (CSV)", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, parse_dates=['Date'])
            st.success("Data loaded successfully!")
    elif data_source == "Generate synthetic data":
        n_points = st.slider("Number of data points", min_value=100, max_value=1000, value=365, step=100)
        if st.button("Generate Data"):
            df = generate_synthetic_data(n_points)
            st.success(f"Generated {n_points} data points successfully!")
    else:
        df = pd.read_csv('data/sample_data.csv', parse_dates=['Date'])
        st.success("Sample data loaded successfully!")

    if 'df' in locals():
        # Data configuration
        target_col = st.selectbox(
            "Select target variable (e.g., Sales)", 
            [col for col in df.columns if col != 'Date']
        )
        media_channels = st.multiselect(
            "Select media channels",
            [col for col in df.columns if col not in [target_col, 'Date']],
            default=[col for col in df.columns if col not in [target_col, 'Date']][:3]
        )

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

    # Run Analysis button
    if st.button("Run Analysis"):
        # Correlation Analysis
        st.header("Correlation Analysis")
        corr_matrix = df[media_channels + [target_col]].corr()
        
        # Heatmap using plotly
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu"
        )
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
        fig = px.bar(
            importance_df, 
            x='Coefficient', 
            y='Channel', 
            orientation='h',
            title='Channel Impact on ' + target_col
        )
        st.plotly_chart(fig, use_container_width=True)

        # Time Series Analysis
        st.header("Time Series Analysis")
        fig = px.line(
            df, 
            x='Date', 
            y=[target_col] + media_channels,
            title=f"Time Series View of {target_col} and Channel Spending"
        )
        st.plotly_chart(fig, use_container_width=True)

        # What-If Analysis
        st.header("What-If Analysis")
        st.write("Adjust channel spending to see the impact on " + target_col)
        
        col1, col2 = st.columns(2)
        
        # Current values and adjustments
        current_values = {}
        adjusted_values = {}
        
        with col1:
            st.subheader("Current Values")
            for channel in media_channels:
                current_values[channel] = df[channel].mean()
                st.metric(channel, f"{current_values[channel]:,.0f}")
        
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
            st.metric("Current Predicted " + target_col, f"{current_pred:,.0f}")
        with col2:
            st.metric(
                "Adjusted Predicted " + target_col, 
                f"{adjusted_pred:,.0f}", 
                delta=f"{adjusted_pred - current_pred:,.0f}"
            )

else:
    st.info("Please select a data source to begin analysis.")
