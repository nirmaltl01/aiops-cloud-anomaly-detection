import pyspark
import sklearn
import tensorflow as tf
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

# Initialize a local Spark Session
spark = SparkSession.builder \
    .appName("SetupTest") \
    .master("local[*]") \
    .getOrCreate()

print("\n--- Environment Check ---")
print(f"PySpark Version : {spark.version}")
print(f"TensorFlow      : {tf.__version__}")
print(f"Scikit-Learn    : {sklearn.__version__}")
print("Status          : All frameworks loaded successfully!\n")

spark.stop()