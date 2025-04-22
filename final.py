import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Transformer Maintenance",
    page_icon="⚡",
    layout="wide"
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
This application monitors power transformers and predicts potential failures using machine learning.
""")

# Function to load data
@st.cache_data
def load_data():
    """Load the transformer data from CSV"""
    if not os.path.exists('transformer_data.csv'):
        st.error("Data file not found. Please run dataset.py first.")
        return None
        
    df = pd.read_csv('transformer_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Function to load the trained model
@st.cache_resource
def load_model():
    """Load the trained model"""
    if not os.path.exists('transformer_model.joblib'):
        st.warning("Model file not found. Please run train.py first.")
        return None
        
    model = joblib.load('transformer_model.joblib')
    return model

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

# Create time series plot
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

# Main application
def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Load model
    model = load_model()
    
    # Create navigation in sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Dashboard", "Data Analysis", "Predictions"]
    )
    
    # Get available transformer IDs
    transformer_ids = sorted(df['transformer_id'].unique())
    
    # Dashboard page
    if page == "Dashboard":
        st.header("Transformer Monitoring Dashboard")
        
        # Dashboard layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Transformer Status")
            
            # Display status for each transformer
            for transformer_id in transformer_ids:
                # Get the most recent data
                latest = df[df['transformer_id'] == transformer_id].iloc[-1]
                
                # Get status
                health_index = latest['health_index']
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
    
    # Data Analysis page
    elif page == "Data Analysis":
        st.header("Data Analysis & Statistics")
        
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
        
        # Correlation analysis
        st.subheader("Parameter Correlations")
        
        # Select columns for correlation
        numeric_cols = ['temperature', 'load_percentage', 'oil_quality', 
                       'dissolved_gas', 'vibration', 'health_index']
        
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
        
        # Status distribution
        st.subheader("Status Distribution")
        
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig = px.pie(
            status_counts, 
            values='Count', 
            names='Status',
            title='Distribution of Transformer Status',
            color='Status',
            color_discrete_map={'Normal': 'green', 'Warning': 'orange', 'Critical': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictions page
    elif page == "Predictions":
        st.header("Transformer Status Predictions")
        
        if model is None:
            st.warning("Model not loaded. Please run train.py to train the model.")
            return
        
        st.write("Use this page to predict transformer status based on sensor readings.")
        
        # Prediction options
        prediction_type = st.radio(
            "Select prediction source:",
            ["Use latest transformer data", "Enter custom values"]
        )
        
        if prediction_type == "Use latest transformer data":
            # Select transformer
            selected_transformer = st.selectbox(
                "Select Transformer",
                transformer_ids,
                key="pred_transformer"
            )
            
            # Get latest data
            latest = df[df['transformer_id'] == selected_transformer].iloc[-1]
            
            # Display current values
            st.subheader("Current Sensor Values:")
            col1, col2, col3 = st.columns(3)
            col1.metric("Temperature", f"{latest['temperature']:.1f}°C")
            col1.metric("Load", f"{latest['load_percentage']:.1f}%")
            col2.metric("Oil Quality", f"{latest['oil_quality']:.1f}")
            col2.metric("Dissolved Gas", f"{latest['dissolved_gas']:.1f} ppm")
            col3.metric("Vibration", f"{latest['vibration']:.3f}")
            col3.metric("Health Index", f"{latest['health_index']:.1f}")
            
            # Create input array for prediction
            input_data = latest[['temperature', 'load_percentage', 'oil_quality', 
                                 'dissolved_gas', 'vibration']].values.reshape(1, -1)
            
        else:  # Custom values
            st.subheader("Enter Sensor Values:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider("Temperature (°C)", 20.0, 120.0, 45.0, 0.1)
                load_percentage = st.slider("Load (%)", 0.0, 100.0, 70.0, 0.1)
                oil_quality = st.slider("Oil Quality", 0.0, 100.0, 90.0, 0.1)
            
            with col2:
                dissolved_gas = st.slider("Dissolved Gas (ppm)", 0.0, 200.0, 50.0, 0.1)
                vibration = st.slider("Vibration", 0.0, 2.0, 0.3, 0.01)
            
            # Create input array for prediction
            input_data = np.array([temperature, load_percentage, oil_quality, 
                                   dissolved_gas, vibration]).reshape(1, -1)
        
        # Make prediction when button is clicked
        if st.button("Predict Status"):
            # Predict status
            status_code = model.predict(input_data)[0]
            status_proba = model.predict_proba(input_data)[0]
            
            # Get the actual classes the model knows about
            model_classes = model.classes_
            
            # Map back to status labels - create a robust mapping
            status_labels = {0: 'Normal', 1: 'Warning', 2: 'Critical'}
            
            # Make sure we have a valid status even if model.classes_ doesn't align with our expectations
            if status_code in status_labels:
                status = status_labels[status_code]
            else:
                status = f"Class {status_code}"  # Fallback for unexpected class codes
            
            # Display prediction
            status_color_map = {'Normal': 'green', 'Warning': 'orange', 'Critical': 'red'}
            status_color = status_color_map.get(status, "#3366CC")  # Default blue for unknown status
            
            # Build probability HTML dynamically based on available classes
            probability_html = "<p>Confidence:</p><ul>"
            for i, class_idx in enumerate(model_classes):
                if i < len(status_proba):
                    class_name = status_labels.get(class_idx, f"Class {class_idx}")
                    probability_html += f"<li>{class_name}: {status_proba[i]:.1%}</li>"
            probability_html += "</ul>"
            
            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; margin:20px 0; 
                        border:2px solid {status_color}; background-color:{status_color}20">
                <h2>Predicted Status: <span style="color:{status_color}">{status}</span></h2>
                {probability_html}
            </div>
            """, unsafe_allow_html=True)
            
            # Display recommendations based on status
            st.subheader("Maintenance Recommendations")
            
            if status == 'Critical':
                st.error("🔴 Urgent maintenance required. Schedule inspection within 48 hours.")
                st.markdown("- Check oil quality and consider oil replacement")
                st.markdown("- Inspect cooling system")
                st.markdown("- Perform dissolved gas analysis (DGA)")
            elif status == 'Warning':
                st.warning("🟠 Maintenance recommended within 2 weeks.")
                st.markdown("- Monitor temperature closely")
                st.markdown("- Check oil level and quality")
                st.markdown("- Inspect for unusual vibrations")
            else:
                st.success("🟢 Transformer in good condition. Follow standard maintenance schedule.")

if __name__ == "__main__":
    main()