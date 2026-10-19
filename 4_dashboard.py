import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set up the web page
st.set_page_config(page_title="Cloud Server Anomaly Detection", layout="wide")
st.title("☁️ Multi-Model Cloud Server Anomaly Detection")
st.markdown("Powered by PySpark, Z-Score, Isolation Forest, & LSTM Autoencoder")

# Load our final processed data
@st.cache_data
def load_data():
    df = pd.read_csv("final_multimodel_telemetry.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# Top Summary Metrics
st.subheader("System Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Telemetry Records", f"{len(df):,}")
col2.metric("Z-Score (Statistical)", int(df['zscore_anomaly'].sum()))
col3.metric("Isolation Forest (ML)", int(df['if_anomaly'].sum()))
col4.metric("🚨 SEVERE ANOMALIES", int(df['severe_anomaly'].sum()))

st.divider()

# Interactive Chart
st.subheader("Interactive Telemetry Analysis")
metric = st.selectbox("Select Server Metric to Visualize:", 
                     ['cpu_usage', 'memory_usage', 'network_out', 'disk_io'])

fig = go.Figure()

# Plot the normal data as a line
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df[metric],
    mode='lines', name=f'Normal Traffic',
    line=dict(color='#1f77b4', width=1)
))

# Overlay the SEVERE anomalies as red X's
severe_anomalies = df[df['severe_anomaly'] == 1]
fig.add_trace(go.Scatter(
    x=severe_anomalies['timestamp'], y=severe_anomalies[metric],
    mode='markers', name='Severe Anomaly',
    marker=dict(color='red', size=10, symbol='x', line=dict(color='black', width=1))
))

fig.update_layout(
    title=f"{metric.replace('_', ' ').title()} over Time",
    xaxis_title="Time",
    yaxis_title="Usage / Volume",
    hovermode="x unified"
)

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# Show the raw anomaly data table
st.subheader("Severe Anomaly Log")
st.dataframe(
    severe_anomalies[['timestamp', 'cpu_usage', 'memory_usage', 'network_out', 'disk_io', 'total_flags']],
    use_container_width=True
)