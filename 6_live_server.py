import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

# Create a directory to hold our live stream data
os.makedirs("live_logs", exist_ok=True)
print("🚀 Live Server Simulator Started. Press Ctrl+C to stop.")
print("Generating live telemetry every 2 seconds...")

file_index = 0
while True:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Inject a severe anomaly ~10% of the time to test the stream
    is_anomaly = np.random.rand() > 0.9
    cpu = np.random.uniform(90, 100) if is_anomaly else np.random.normal(45, 10)
    
    data = {
        'timestamp': [timestamp],
        'cpu_usage': [max(0, cpu)],
        'memory_usage': [max(0, np.random.normal(60, 5))],
        'network_in': [max(0, np.random.normal(100, 20))],
        'network_out': [max(0, np.random.normal(90, 15))],
        'disk_io': [max(0, np.random.normal(50, 10))]
    }
    
    df = pd.DataFrame(data)
    # Write to a new CSV file to simulate an incoming data stream
    filename = f"live_logs/log_{file_index:04d}.csv"
    df.to_csv(filename, index=False)
    
    status = "🚨 SPIKE INJECTED" if is_anomaly else "✅ Normal"
    print(f"[{timestamp}] Sent telemetry -> CPU: {cpu:.1f}% | {status}")
    
    file_index += 1
    time.sleep(2)
    