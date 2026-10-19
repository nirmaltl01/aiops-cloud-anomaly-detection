from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, abs as spark_abs, when

def run_zscore_analysis():
    print("Initializing PySpark...")
    spark = SparkSession.builder \
        .appName("ZScoreAnomalyDetection") \
        .master("local[*]") \
        .getOrCreate()

    # 1. Load the data using PySpark
    print("Loading telemetry data...")
    df = spark.read.csv("cloud_telemetry.csv", header=True, inferSchema=True)
    
    # We will analyze 'cpu_usage' for basic statistical spikes
    metric = "cpu_usage"
    
    # 2. Calculate Mean and Standard Deviation
    stats = df.select(
        mean(col(metric)).alias("mean"), 
        stddev(col(metric)).alias("stddev")
    ).collect()[0]
    
    mean_val = stats["mean"]
    std_val = stats["stddev"]
    print(f"Dataset CPU Mean: {mean_val:.2f}, StdDev: {std_val:.2f}")

    # 3. Calculate Z-Score and flag anomalies (Z-Score > 3 is a standard threshold)
    print("Calculating Z-Scores...")
    df_with_z = df.withColumn(
        "z_score", 
        spark_abs(col(metric) - mean_val) / std_val
    )
    
    df_flagged = df_with_z.withColumn(
        "zscore_anomaly", 
        when(col("z_score") > 3.0, 1).otherwise(0)
    )

    # 4. Show results
    total_anomalies = df_flagged.filter(col("zscore_anomaly") == 1).count()
    print(f"\nFound {total_anomalies} anomalies using Z-Score.")
    
    print("\nTop 5 Detected CPU Spikes:")
    df_flagged.filter(col("zscore_anomaly") == 1) \
        .select("timestamp", "cpu_usage", "z_score", "zscore_anomaly") \
        .orderBy(col("z_score").desc()) \
        .show(5)

    # Convert back to pandas for our next models
    df_pandas = df_flagged.toPandas()
    df_pandas.to_csv("telemetry_with_zscore.csv", index=False)
    
    spark.stop()
    print("Pipeline step 1 complete. Saved to 'telemetry_with_zscore.csv'")

if __name__ == "__main__":
    run_zscore_analysis()