import pandas as pd
import numpy as np
import datetime

def generate_sample_data(num_transformers=3, days=90, filename='transformer_data.csv'):
    """Generate synthetic transformer data for demonstration and save to CSV"""
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
    
    # Calculate health index for each transformer
    combined_df['health_index'] = combined_df.apply(calculate_health_index, axis=1)
    
    # Save to CSV
    combined_df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    return combined_df

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

if __name__ == "__main__":
    # Generate the dataset and save to CSV
    generate_sample_data(num_transformers=3, days=90, filename='transformer_data.csv')