import pandas as pd
import numpy as np

def generate_telemetry_data(num_samples=10000):
    print("Generating normal cloud telemetry data...")
    np.random.seed(42)
    
    # Base normal behavior (e.g., standard server load)
    data = {
        'timestamp': pd.date_range(start='2026-10-10', periods=num_samples, freq='min'),
        'cpu_usage': np.random.normal(45, 10, num_samples),      # Mean 45%, StdDev 10
        'memory_usage': np.random.normal(60, 5, num_samples),    # Mean 60%, StdDev 5
        'network_in': np.random.normal(100, 20, num_samples),    # MB/s
        'network_out': np.random.normal(90, 15, num_samples),    # MB/s
        'disk_io': np.random.normal(50, 10, num_samples),        # MB/s
        'label': 0  # 0 means normal
    }
    
    df = pd.DataFrame(data)
    
    # Inject Anomalies
    print("Injecting anomalies (CPU spikes, Memory leaks, Network Drops)...")
    
    # 1. CPU Spikes (Statistical anomaly - good for Z-Score)
    spike_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[spike_indices, 'cpu_usage'] = np.random.uniform(90, 100, size=50)
    df.loc[spike_indices, 'label'] = 1
    
    # 2. Memory Leaks (Temporal anomaly - good for LSTM)
    # Gradual increase over 100 minutes
    leak_start = 5000
    df.loc[leak_start:leak_start+99, 'memory_usage'] = np.linspace(65, 98, 100)
    df.loc[leak_start:leak_start+99, 'label'] = 1
    
    # 3. Multivariate Anomaly (good for Isolation Forest)
    # CPU is low, but Network and Disk are maxed out (unusual combination)
    multi_indices = np.random.choice(df.index[:4900], size=30, replace=False)
    df.loc[multi_indices, 'cpu_usage'] = np.random.uniform(5, 15, size=30)
    df.loc[multi_indices, 'network_out'] = np.random.uniform(200, 250, size=30)
    df.loc[multi_indices, 'disk_io'] = np.random.uniform(150, 200, size=30)
    df.loc[multi_indices, 'label'] = 1

    # Ensure no negative values
    numeric_cols = ['cpu_usage', 'memory_usage', 'network_in', 'network_out', 'disk_io']
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    
    return df

if __name__ == "__main__":
    df = generate_telemetry_data()
    df.to_csv("cloud_telemetry.csv", index=False)
    print("Data saved to 'cloud_telemetry.csv'")
    print(f"Total Records: {len(df)}")
    print(f"Total Anomalies Injected: {df['label'].sum()}")