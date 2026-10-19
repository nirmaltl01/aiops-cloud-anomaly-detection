# AIOps Cloud Server Anomaly Detection Pipeline

## Overview
This project is an end-to-end AIOps (Artificial Intelligence for IT Operations) pipeline designed to ingest, process, and analyze cloud server telemetry. It detects system degradations (such as memory leaks, network drops, and CPU spikes) using a multi-model consensus engine to drastically reduce false-positive alerts.

## Architecture
The system operates in two distinct modes:

1. **Batch Processing & AI Consensus (Files 1-5):** Processes historical server logs using PySpark and evaluates them through three distinct algorithms:
   - **Z-Score:** Statistical baseline analysis.
   - **Isolation Forest:** Multivariate machine learning for complex metric relationships.
   - **LSTM Autoencoder:** Temporal deep learning to reconstruct sequence data and catch gradual degradation.
   - **Consensus Engine:** A voting mechanism that aggregates predictions to ensure high-fidelity alerts.
   - **Dashboard:** An interactive Streamlit UI for visual analytics.

2. **Real-Time Streaming (Files 6-7):** A custom Python telemetry simulator continuously generates synthetic hardware metrics. A PySpark Structured Streaming engine ingests these micro-batches in real time to perform live statistical anomaly detection on high-velocity data.

## Technology Stack
* **Data Engineering:** Apache PySpark (Structured Streaming & DataFrames)
* **Machine Learning / AI:** TensorFlow (Keras), Scikit-Learn
* **Visualization:** Streamlit, Plotly, Seaborn, Matplotlib
* **Languages:** Python, Pandas, NumPy

## Future Scope (Enterprise Cloud Deployment)
While this proof-of-concept utilizes a local synthetic data generator and directory-based streaming, the architecture is designed to be cloud-agnostic. In a production deployment, the data ingestion layer would integrate directly with **Apache Kafka** (or Amazon MSK), and the PySpark processing engine would scale via **AWS EMR** (Elastic MapReduce) clusters.
