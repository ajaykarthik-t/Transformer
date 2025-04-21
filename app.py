import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import time

# Set page configuration with a clean layout
st.set_page_config(
    page_title="Transformer Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add some basic styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .warning {
        color: #ff0000;
        font-weight: bold;
    }
    .normal {
        color: #4CAF50;
        font-weight: bold;
    }
    .caution {
        color: #FFA500;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Application title
st.title("⚡ Power Transformer Predictive Maintenance")
st.markdown("""
This application helps monitor and predict failures in power transformers using AI and machine learning.
It analyzes sensor data to detect anomalies and provide maintenance recommendations.
""")

# Create navigation in sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Dashboard", "Data Analysis", "Model Training", "Maintenance Planner"]
)

# Function to generate sample transformer data
def generate_sample_data(num_transformers=3, days=90):
    """Generate synthetic transformer data for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    # Create date range (hourly data for the specified number of days)
    date_range = pd.date_range(end=datetime.datetime.now(), periods=days*24, freq='1H')
    
    transformers = []
    
    for transformer_id in range(1, num_transformers+1):
        # Base parameters (with some variation between transformers)
        base_temp = 45 + np.random.randint(-5, 5)
        base_load = 70 + np.random.randint(-10, 10)
        base_oil_quality = 95 + np.random.randint(-5, 5)
        
        # Create data with daily and seasonal variations
        temps = []
        loads = []
        oil_qualities = []
        dissolved_gases = []
        vibrations = []
        statuses = []
        
        # Generate data for each hour
        for i, timestamp in enumerate(date_range):
            # Daily variation (higher during daytime)
            hour_of_day = timestamp.hour
            daily_effect = 5 * np.sin(2 * np.pi * hour_of_day / 24)
            
            # Random variation
            random_var = np.random.normal(0, 1)
            
            # Calculate parameter values
            temp = base_temp + daily_effect + random_var * 2
            load = base_load + daily_effect + random_var * 5
            
            # Oil quality decreases slowly over time
            oil_degradation = i / (len(date_range) * 2) * 10
            oil_quality = base_oil_quality - oil_degradation + np.random.normal(0, 0.2)
            
            # Dissolved gas increases with temperature
            gas = 50 + temp/10 + np.random.normal(0, 5)
            
            # Vibration relates to load
            vibration = 0.2 + load/1000 + np.random.normal(0, 0.05)
            
            # Default status is normal
            status = "Normal"
            
            # Add to lists
            temps.append(temp)
            loads.append(load)
            oil_qualities.append(oil_quality)
            dissolved_gases.append(gas)
            vibrations.append(vibration)
            statuses.append(status)
        
        # Add some anomalies to make the data interesting
        # 1. Add a temperature spike for one transformer
        if transformer_id == 1:
            spike_start = int(len(temps) * 0.7)  # At 70% through the time period
            spike_length = 48  # 48 hours (2 days)
            for i in range(spike_start, min(spike_start + spike_length, len(temps))):
                temps[i] += 20 + np.random.random() * 5  # Significant temperature increase
                if temps[i] > 85:
                    statuses[i] = "Warning"
                if temps[i] > 95:
                    statuses[i] = "Critical"
        
        # 2. Add oil quality degradation for another transformer
        if transformer_id == 2:
            degradation_start = int(len(temps) * 0.5)  # At 50% through the time period
            for i in range(degradation_start, len(temps)):
                progress = (i - degradation_start) / (len(temps) - degradation_start)
                oil_qualities[i] -= progress * 30  # Progressive degradation
                dissolved_gases[i] += progress * 70  # Gas increases
                
                # Update status based on severity
                if oil_qualities[i] < 50:
                    statuses[i] = "Critical"
                elif oil_qualities[i] < 70:
                    statuses[i] = "Warning"
        
        # Create DataFrame for this transformer
        df = pd.DataFrame({
            'timestamp': date_range,
            'transformer_id': transformer_id,
            'temperature': temps,
            'load_percentage': loads,
            'oil_quality': oil_qualities,
            'dissolved_gas': dissolved_gases,
            'vibration': vibrations,
            'status': statuses
        })
        
        transformers.append(df)
    
    # Combine all transformer data
    combined_df = pd.concat(transformers, ignore_index=True)
    
    # Add useful time-based features
    combined_df['day_of_week'] = combined_df['timestamp'].dt.dayofweek
    combined_df['hour_of_day'] = combined_df['timestamp'].dt.hour
    
    return combined_df

# Calculate health index for transformers
def calculate_health_index(row):
    """Calculate overall transformer health index (0-100)"""
    # Weights for different parameters
    weights = {
        'temperature': 0.2,      # 20% weight
        'load_percentage': 0.15, # 15% weight
        'oil_quality': 0.4,      # 40% weight - most important
        'dissolved_gas': 0.15,   # 15% weight
        'vibration': 0.1         # 10% weight
    }
    
    # Convert each parameter to a 0-100 score
    temp_score = max(0, 100 - max(0, (row['temperature'] - 40) * 2))
    load_score = max(0, 100 - max(0, (row['load_percentage'] - 75) * 4))
    oil_score = row['oil_quality']  # Oil quality is already on a 0-100 scale
    gas_score = max(0, 100 - row['dissolved_gas'] / 2)
    vibration_score = max(0, 100 - row['vibration'] * 50)
    
    # Calculate weighted average
    health_index = (
        weights['temperature'] * temp_score +
        weights['load_percentage'] * load_score +
        weights['oil_quality'] * oil_score +
        weights['dissolved_gas'] * gas_score +
        weights['vibration'] * vibration_score
    )
    
    return health_index

# Get health status based on index
def get_health_status(health_index):
    """Get health status and color based on health index"""
    if health_index >= 85:
        return "Normal", "#4CAF50"  # Green
    elif health_index >= 70:
        return "Good", "#8BC34A"    # Light Green
    elif health_index >= 50:
        return "Caution", "#FFC107" # Amber
    elif health_index >= 30:
        return "Warning", "#FF9800" # Orange
    else:
        return "Critical", "#F44336" # Red

# Create time series plot for a specific metric
def create_time_series_plot(df, transformer_id, column, title, y_label, color='blue'):
    """Create a time series plot for a specific transformer and metric"""
    # Filter data for the selected transformer
    transformer_df = df[df['transformer_id'] == transformer_id].copy()
    
    # Create the plot using Plotly
    fig = px.line(
        transformer_df, 
        x='timestamp', 
        y=column,
        title=title,
        labels={'timestamp': 'Time', column: y_label}
    )
    
    # Apply some styling to the plot
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        title_x=0.5,
        plot_bgcolor='white'
    )
    
    # Add status indicators as colored points
    if 'status' in transformer_df.columns:
        # Define color mapping for status
        color_map = {
            'Normal': 'green',
            'Warning': 'orange',
            'Critical': 'red'
        }
        
        # Add colored points for each status type
        for status, color in color_map.items():
            status_df = transformer_df[transformer_df['status'] == status]
            if not status_df.empty:
                fig.add_scatter(
                    x=status_df['timestamp'],
                    y=status_df[column],
                    mode='markers',
                    marker=dict(color=color, size=8),
                    name=status
                )
    
    return fig

# Function to detect anomalies
def detect_anomalies(df, transformer_id):
    """Detect anomalies in transformer data using Isolation Forest"""
    # Filter for the specific transformer
    transformer_df = df[df['transformer_id'] == transformer_id].copy()
    
    # Select features for anomaly detection
    features = ['temperature', 'load_percentage', 'oil_quality', 
                'dissolved_gas', 'vibration']
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(transformer_df[features])
    
    # Train isolation forest (unsupervised anomaly detection)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(scaled_features)
    
    # Predict anomalies
    transformer_df['anomaly_score'] = iso_forest.decision_function(scaled_features)
    transformer_df['is_anomaly'] = np.where(iso_forest.predict(scaled_features) == -1, 1, 0)
    
    return transformer_df

# Train a simple random forest model
def train_simple_model(df):
    """Train a Random Forest model to predict transformer status"""
    # Convert status to numerical values
    status_map = {'Normal': 0, 'Warning': 1, 'Critical': 2}
    df['status_code'] = df['status'].map(status_map)
    
    # Select features and target
    features = ['temperature', 'load_percentage', 'oil_quality', 
                'dissolved_gas', 'vibration']
    X = df[features]
    y = df['status_code']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, accuracy, feature_importance

# Generate data when the app starts
df = generate_sample_data(num_transformers=3, days=90)

# Dashboard page
if page == "Dashboard":
    st.header("Transformer Monitoring Dashboard")
    
    # Get available transformer IDs
    transformer_ids = sorted(df['transformer_id'].unique())
    
    # Dashboard layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Transformer Status")
        
        # Calculate latest status for each transformer
        for transformer_id in transformer_ids:
            # Get the most recent data
            latest = df[df['transformer_id'] == transformer_id].iloc[-1]
            
            # Calculate health index
            health_index = calculate_health_index(latest)
            status, color = get_health_status(health_index)
            
            # Display status card
            st.markdown(f"""
            <div style="padding:10px; border-radius:5px; margin-bottom:10px; border:1px solid #ddd">
                <h4>Transformer #{transformer_id}</h4>
                <p>Health Index: <span style="color:{color}">{health_index:.1f}</span></p>
                <p>Status: <span style="color:{color}">{status}</span></p>
                <p>Temperature: {latest['temperature']:.1f}°C</p>
                <p>Oil Quality: {latest['oil_quality']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Add transformer selector
        selected_transformer = st.selectbox(
            "Select Transformer for Detailed View",
            transformer_ids,
            format_func=lambda x: f"Transformer #{x}"
        )
        
        # Display time series for selected transformer
        st.subheader(f"Transformer #{selected_transformer} - Monitoring Data")
        
        # Create tabs for different metrics
        tab1, tab2 = st.tabs(["Temperature & Load", "Oil Quality & Gas"])
        
        with tab1:
            temp_chart = create_time_series_plot(
                df, selected_transformer, 'temperature', 
                f"Temperature History", 
                "Temperature (°C)", "red"
            )
            st.plotly_chart(temp_chart, use_container_width=True)
            
            load_chart = create_time_series_plot(
                df, selected_transformer, 'load_percentage', 
                f"Load History", 
                "Load (%)", "blue"
            )
            st.plotly_chart(load_chart, use_container_width=True)
        
        with tab2:
            oil_chart = create_time_series_plot(
                df, selected_transformer, 'oil_quality', 
                f"Oil Quality History", 
                "Oil Quality Index", "green"
            )
            st.plotly_chart(oil_chart, use_container_width=True)
            
            gas_chart = create_time_series_plot(
                df, selected_transformer, 'dissolved_gas', 
                f"Dissolved Gas Analysis", 
                "Gas (ppm)", "purple"
            )
            st.plotly_chart(gas_chart, use_container_width=True)
    
    # Recommendations section
    st.subheader("Maintenance Recommendations")
    
    # Get the latest health index for the selected transformer
    latest = df[df['transformer_id'] == selected_transformer].iloc[-1]
    health_index = calculate_health_index(latest)
    
    # Display recommendations based on health index
    if health_index < 50:
        st.error("🔴 Urgent maintenance required. Schedule inspection within 48 hours.")
        st.markdown("- Check oil quality and consider oil replacement")
        st.markdown("- Inspect cooling system")
        st.markdown("- Perform dissolved gas analysis (DGA)")
    elif health_index < 70:
        st.warning("🟠 Maintenance recommended within 2 weeks.")
        st.markdown("- Monitor temperature closely")
        st.markdown("- Check oil level and quality")
        st.markdown("- Inspect for unusual vibrations")
    else:
        st.success("🟢 Transformer in good condition. Follow standard maintenance schedule.")

# Data Analysis page
elif page == "Data Analysis":
    st.header("Data Analysis & Anomaly Detection")
    
    # Show data overview
    st.subheader("Dataset Overview")
    
    # Get some basic stats
    transformer_count = df['transformer_id'].nunique()
    time_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    data_points = len(df)
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Transformers", transformer_count)
    col2.metric("Time Range", time_range)
    col3.metric("Total Data Points", f"{data_points:,}")
    
    # Show data sample
    with st.expander("View Data Sample"):
        st.dataframe(df.head(10))
    
    # Anomaly detection section
    st.subheader("Anomaly Detection")
    
    # Select transformer for analysis
    selected_transformer = st.selectbox(
        "Select Transformer",
        sorted(df['transformer_id'].unique()),
        key="anomaly_transformer"
    )
    
    # Run anomaly detection
    anomaly_results = detect_anomalies(df, selected_transformer)
    anomaly_count = anomaly_results['is_anomaly'].sum()
    
    st.write(f"Detected {anomaly_count} potential anomalies out of {len(anomaly_results)} data points.")
    
    # Plot temperature with anomalies highlighted
    fig = go.Figure()
    
    # Add normal points
    normal_df = anomaly_results[anomaly_results['is_anomaly'] == 0]
    fig.add_trace(
        go.Scatter(
            x=normal_df['timestamp'],
            y=normal_df['temperature'],
            mode='lines',
            name='Normal',
            line=dict(color='blue', width=1)
        )
    )
    
    # Add anomaly points
    anomaly_df = anomaly_results[anomaly_results['is_anomaly'] == 1]
    fig.add_trace(
        go.Scatter(
            x=anomaly_df['timestamp'],
            y=anomaly_df['temperature'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=8)
        )
    )
    
    fig.update_layout(
        title=f"Temperature with Detected Anomalies - Transformer #{selected_transformer}",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top anomalies
    if anomaly_count > 0:
        st.subheader("Top Anomalies")
        
        # Sort anomalies by score (most anomalous first)
        top_anomalies = anomaly_df.sort_values('anomaly_score').head(5)
        
        for idx, anomaly in top_anomalies.iterrows():
            st.markdown(f"""
            <div style="padding:10px; background-color:#fff3f3; border-radius:5px; margin-bottom:10px; border:1px solid #ffcccb">
                <p><strong>Time: {anomaly['timestamp'].strftime('%Y-%m-%d %H:%M')}</strong></p>
                <p>Temperature: {anomaly['temperature']:.1f}°C | Load: {anomaly['load_percentage']:.1f}%</p>
                <p>Oil Quality: {anomaly['oil_quality']:.1f} | Gas: {anomaly['dissolved_gas']:.1f} ppm</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Correlation analysis
    st.subheader("Parameter Correlations")
    
    # Select columns for correlation
    numeric_cols = ['temperature', 'load_percentage', 'oil_quality', 
                   'dissolved_gas', 'vibration']
    
    # Calculate correlation
    corr = df[numeric_cols].corr()
    
    # Plot correlation heatmap
    fig = px.imshow(
        corr, 
        text_auto=True, 
        color_continuous_scale='RdBu_r',
        title="Parameter Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

# Model Training page
elif page == "Model Training":
    st.header("Predictive Model Training")
    
    st.write("""
    This page allows you to train a machine learning model to predict transformer status
    based on sensor data. The model can help identify potential issues before they become critical.
    """)
    
    # Model parameters
    st.subheader("Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
        max_depth = st.slider("Max Depth", 5, 30, 10, 5)
    
    with col2:
        # Feature selection
        st.write("Features for Training:")
        use_temperature = st.checkbox("Temperature", value=True)
        use_load = st.checkbox("Load Percentage", value=True)
        use_oil = st.checkbox("Oil Quality", value=True)
        use_gas = st.checkbox("Dissolved Gas", value=True)
        use_vibration = st.checkbox("Vibration", value=True)
    
    # Create feature list based on selections
    selected_features = []
    if use_temperature:
        selected_features.append('temperature')
    if use_load:
        selected_features.append('load_percentage')
    if use_oil:
        selected_features.append('oil_quality')
    if use_gas:
        selected_features.append('dissolved_gas')
    if use_vibration:
        selected_features.append('vibration')
    
    # Ensure at least one feature is selected
    if len(selected_features) == 0:
        st.warning("Please select at least one feature for training.")
        selected_features = ['temperature']  # Default to temperature if nothing selected
    
    # Train model button
    if st.button("Train Model"):
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update status
        status_text.text("Preparing data...")
        progress_bar.progress(20)
        
        # Filter dataframe to only include selected features
        model_df = df.copy()
        
        # Update status
        time.sleep(0.5)  # Simulate processing time
        status_text.text("Training model...")
        progress_bar.progress(50)
        
        # Train model
        model, accuracy, feature_importance = train_simple_model(model_df)
        
        # Update status
        time.sleep(0.5)  # Simulate processing time
        status_text.text("Evaluating model...")
        progress_bar.progress(80)
        
        # Complete
        time.sleep(0.5)  # Simulate processing time
        progress_bar.progress(100)
        status_text.text("Model training complete!")
        
        # Show results
        st.success(f"Model trained successfully with {accuracy:.2%} accuracy!")
        
        # Show feature importance
        st.subheader("Feature Importance")
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance in Predicting Transformer Status"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of results
        st.write("""
        **What this means:**
        
        The chart above shows how important each feature is for predicting transformer status.
        Higher values indicate that the parameter has a stronger influence on the model's predictions.
        
        In a real deployment, this model would be used to:
        - Predict potential failures before they occur
        - Identify which parameters to monitor most closely
        - Recommend maintenance actions based on sensor data
        """)

# Maintenance Planner page
elif page == "Maintenance Planner":
    st.header("Maintenance Planning & Scheduling")
    
    # Get transformers that need maintenance
    transformer_health = []
    for transformer_id in df['transformer_id'].unique():
        latest = df[df['transformer_id'] == transformer_id].iloc[-1]
        health_index = calculate_health_index(latest)
        status, color = get_health_status(health_index)
        
        transformer_health.append({
            'transformer_id': transformer_id,
            'health_index': health_index,
            'status': status,
            'color': color,
            'temperature': latest['temperature'],
            'oil_quality': latest['oil_quality'],
            'dissolved_gas': latest['dissolved_gas']
        })
    
    # Sort by health index (worst first)
    transformer_health.sort(key=lambda x: x['health_index'])
    
    # Display maintenance priorities
    st.subheader("Maintenance Priorities")
    
    for transformer in transformer_health:
        priority = "High" if transformer['health_index'] < 50 else "Medium" if transformer['health_index'] < 70 else "Low"
        priority_color = "#F44336" if priority == "High" else "#FFA500" if priority == "Medium" else "#4CAF50"
        
        # Suggested timeframe
        timeframe = "Within 48 hours" if priority == "High" else "Within 2 weeks" if priority == "Medium" else "Regular schedule"
        
        st.markdown(f"""
        <div style="padding:15px; border-radius:5px; margin-bottom:15px; border:1px solid {transformer['color']}; background-color:{transformer['color']}20">
            <h3>Transformer #{transformer['transformer_id']}</h3>
            <p>Health Index: <strong style="color:{transformer['color']}">{transformer['health_index']:.1f}</strong></p>
            <p>Status: <strong style="color:{transformer['color']}">{transformer['status']}</strong></p>
            <p>Priority: <strong style="color:{priority_color}">{priority}</strong></p>
            <p>Recommended Timeframe: <strong>{timeframe}</strong></p>
            <p>Key Concerns:</p>
            <ul>
                {"<li>High temperature ("+str(transformer['temperature'])+f"°C)</li>" if transformer['temperature'] > 75 else ""}
                {"<li>Low oil quality ("+str(transformer['oil_quality'])+f"%)</li>" if transformer['oil_quality'] < 70 else ""}
                {"<li>High dissolved gas ("+str(transformer['dissolved_gas'])+f" ppm)</li>" if transformer['dissolved_gas'] > 80 else ""}
                {"<li>General maintenance</li>" if transformer['temperature'] <= 75 and transformer['oil_quality'] >= 70 and transformer['dissolved_gas'] <= 80 else ""}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Maintenance scheduling
    st.subheader("Maintenance Scheduler")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Form for scheduling maintenance
        st.write("Schedule New Maintenance")
        
        # Select transformer
        maintenance_transformer = st.selectbox(
            "Select Transformer",
            sorted(df['transformer_id'].unique()),
            format_func=lambda x: f"Transformer #{x}"
        )
        
        # Maintenance type
        maintenance_type = st.selectbox(
            "Maintenance Type",
            ["Routine Inspection", "Oil Analysis", "Thermal Imaging", "Complete Overhaul", "Emergency Repair"]
        )
        
        # Date and team
        maintenance_date = st.date_input("Scheduled Date", datetime.datetime.now() + datetime.timedelta(days=7))
        maintenance_team = st.selectbox(
            "Assign Team",
            ["Team Alpha", "Team Beta", "Team Gamma", "External Contractor"]
        )
        
        # Notes
        notes = st.text_area("Notes", "Regular maintenance as per schedule.")
        
        # Submit button
        if st.button("Schedule Maintenance"):
            st.success(f"✅ Maintenance scheduled for Transformer #{maintenance_transformer} on {maintenance_date.strftime('%Y-%m-%d')}!")
    
    with col2:
        # Show scheduled maintenance (sample data)
        st.write("Upcoming Maintenance")
        
        # Sample scheduled maintenance
        scheduled = [
            {"id": 1, "transformer": 2, "type": "Oil Analysis", "date": "2025-04-10", "team": "Team Alpha"},
            {"id": 2, "transformer": 1, "type": "Emergency Repair", "date": "2025-03-29", "team": "Team Beta"},
            {"id": 3, "transformer": 3, "type": "Routine Inspection", "date": "2025-04-15", "team": "Team Gamma"}
        ]
        
        for item in scheduled:
            st.markdown(f"""
            <div style="padding:10px; background-color:#f8f9fa; border-radius:5px; margin-bottom:10px; border:1px solid #ddd">
                <p><strong>Transformer #{item['transformer']}</strong> - {item['date']}</p>
                <p>{item['type']} - {item['team']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Placeholder for a maintenance calendar
        st.write("Maintenance Calendar")
        st.image("https://via.placeholder.com/400x300?text=Maintenance+Calendar", use_column_width=True)