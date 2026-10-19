import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

def run_isolation_forest():
    print("Loading Z-Score processed data...")
    # Load the data we saved from the previous step
    df = pd.read_csv("telemetry_with_zscore.csv")
    
    # We want the model to look at how all these metrics interact simultaneously
    features = ['cpu_usage', 'memory_usage', 'network_in', 'network_out', 'disk_io']
    X = df[features]
    
    print("Training Isolation Forest Model (Unsupervised ML)...")
    # contamination=0.01 means we expect roughly 1% of our data to be anomalies
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    
    # Train the model and get predictions in one step
    # Returns -1 for anomalies, 1 for normal
    predictions = iso_forest.fit_predict(X)
    
    # Convert predictions: -1 to 1 (anomaly), 1 to 0 (normal)
    df['if_anomaly'] = [1 if x == -1 else 0 for x in predictions]
    
    total_anomalies = df['if_anomaly'].sum()
    print(f"\nFound {total_anomalies} multi-dimensional anomalies using Isolation Forest.")
    
    # Let's look at the anomalies the Isolation Forest found that Z-Score missed
    unique_if_anomalies = df[(df['if_anomaly'] == 1) & (df['zscore_anomaly'] == 0)]
    print(f"Number of anomalies caught ONLY by Isolation Forest: {len(unique_if_anomalies)}")
    
    if len(unique_if_anomalies) > 0:
        print("\nSample of anomalies missed by Z-Score but caught by ML:")
        print(unique_if_anomalies[['cpu_usage', 'network_out', 'disk_io']].head())
    
    # Save for the final step
    df.to_csv("telemetry_with_ml.csv", index=False)
    print("\nPipeline step 2 complete. Saved to 'telemetry_with_ml.csv'")

if __name__ == "__main__":
    run_isolation_forest()