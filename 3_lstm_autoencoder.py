import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

# Suppress TF warnings for cleaner terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def create_sequences(X, time_steps=10):
    """Convert standard data into time-series sequences"""
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
    return np.array(Xs)

def run_lstm_model():
    print("Loading ML processed data...")
    df = pd.read_csv("telemetry_with_ml.csv")
    
    features = ['cpu_usage', 'memory_usage', 'network_in', 'network_out', 'disk_io']
    
    # 1. Scale the data (Neural Networks need data between 0 and 1)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    
    # 2. Create Time Sequences (Looking at 10 minutes at a time)
    TIME_STEPS = 10
    print(f"Creating time sequences (Window: {TIME_STEPS} minutes)...")
    X_seq = create_sequences(df_scaled, TIME_STEPS)
    
    # 3. Build the LSTM Autoencoder
    print("Building and training LSTM Autoencoder (This might take a minute)...")
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2])),
        RepeatVector(X_seq.shape[1]),
        LSTM(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(X_seq.shape[2]))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model (Keep epochs low for quick testing)
    model.fit(X_seq, X_seq, epochs=3, batch_size=64, validation_split=0.1, verbose=1)
    
    # 4. Predict and Calculate Reconstruction Error
    print("\nCalculating reconstruction errors to find temporal anomalies...")
    X_pred = model.predict(X_seq, verbose=0)
    
    # Calculate Mean Squared Error (MSE) for each sequence
    mse = np.mean(np.power(X_seq - X_pred, 2), axis=(1, 2))
    
    # Pad the beginning of the array with zeros to match the original dataframe length
    padded_mse = np.pad(mse, (TIME_STEPS, 0), 'constant', constant_values=0)
    df['lstm_error'] = padded_mse
    
    # Flag top 2% of errors as anomalies
    threshold = np.percentile(mse, 98) 
    df['lstm_anomaly'] = [1 if error > threshold else 0 for error in df['lstm_error']]
    
    total_lstm = df['lstm_anomaly'].sum()
    print(f"\nFound {total_lstm} temporal anomalies using LSTM.")
    
    # Let's see if it caught our memory leak (around index 5000-5100)
    memory_leak_caught = df.loc[5050:5099, 'lstm_anomaly'].sum()
    print(f"Number of memory leak minutes caught by LSTM: {memory_leak_caught}/50")
    
    # Final Multi-Model Assessment
    # An event is a severe anomaly if 2 or more models flag it
    df['total_flags'] = df['zscore_anomaly'] + df['if_anomaly'] + df['lstm_anomaly']
    df['severe_anomaly'] = df['total_flags'].apply(lambda x: 1 if x >= 2 else 0)
    
    print(f"\nSEVERE ANOMALIES (Flagged by multiple AI models): {df['severe_anomaly'].sum()}")
    
    df.to_csv("final_multimodel_telemetry.csv", index=False)
    print("\nPipeline step 3 complete! Final data saved to 'final_multimodel_telemetry.csv'")

if __name__ == "__main__":
    run_lstm_model()