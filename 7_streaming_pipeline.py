import os
# Inject Hadoop paths dynamically so Windows doesn't crash
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['PATH'] += os.pathsep + 'C:\\hadoop\\bin'
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, abs as spark_abs, when, lit
import warnings

warnings.filterwarnings('ignore')

print("Initializing PySpark Structured Streaming...")
spark = SparkSession.builder \
    .appName("RealTimeAnomalyDetection") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# In streaming, PySpark requires a strict schema definition upfront
schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("cpu_usage", DoubleType(), True),
    StructField("memory_usage", DoubleType(), True),
    StructField("network_in", DoubleType(), True),
    StructField("network_out", DoubleType(), True),
    StructField("disk_io", DoubleType(), True)
])

print("📡 Listening for live server telemetry in 'live_logs/'...")

# 1. Read the streaming data
streaming_df = spark.readStream \
    .schema(schema) \
    .option("header", True) \
    .option("maxFilesPerTrigger", 1) \
    .csv("live_logs/")

# 2. Apply Real-Time Transformation (Streaming Z-Score)
# We use the historical mean (45.13) and stddev (10.75) we learned from our batch analysis
historical_mean = 45.13
historical_std = 10.75

processed_stream = streaming_df.withColumn(
    "z_score", 
    spark_abs(col("cpu_usage") - lit(historical_mean)) / lit(historical_std)
).withColumn(
    "ALERT", 
    when(col("z_score") > 3.0, "🚨 CRITICAL ANOMALY").otherwise("OK")
)

# 3. Output the stream to the console in real-time
query = processed_stream.select("timestamp", "cpu_usage", "z_score", "ALERT") \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .trigger(processingTime="2 seconds") \
    .start()

query.awaitTermination()
