import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(data_file='transformer_data.csv', model_file='transformer_model.joblib'):
    """Train a Random Forest model to predict transformer status"""
    
    # Load the data
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert status to numerical values
    status_map = {'Normal': 0, 'Warning': 1, 'Critical': 2}
    df['status_code'] = df['status'].map(status_map)
    
    # Select features and target
    features = ['temperature', 'load_percentage', 'oil_quality', 
                'dissolved_gas', 'vibration']
    X = df[features]
    y = df['status_code']
    
    # Split data
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2%}")
    
    # Get unique classes in the test set
    unique_classes = sorted(set(y_test) | set(y_pred))
    class_labels = ['Normal', 'Warning', 'Critical']
    
    # Only use target_names that actually appear in the data
    present_classes = [class_labels[i] for i in unique_classes]
    
    print("\nClassification Report:")
    try:
        print(classification_report(y_test, y_pred, target_names=present_classes))
    except ValueError as e:
        print(f"Error generating classification report: {e}")
        print("Generating report without target_names...")
        print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save the model
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")
    
    return model

if __name__ == "__main__":
    # Train the model on the generated dataset
    train_model(data_file='transformer_data.csv', model_file='transformer_model.joblib')