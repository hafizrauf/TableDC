import sqlite3
import pandas as pd
import os
from tqdm import tqdm


os.makedirs('benchmark_parquets', exist_ok=True)
os.makedirs('groundtruth_parquets', exist_ok=True)


benchmark_db = 'benchmark.sqlite'
groundtruth_db = 'groundtruth.sqlite'

# Function to convert database to Parquet files
def db_to_parquet(db_path, output_folder):
    # Connect to the SQLite
    connection = sqlite3.connect(db_path)

    cursor = connection.cursor()

    # Get a list of all tables 
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # For each table, select all records and write to Parquet
    for table in tqdm(tables, desc="Extracting tables"):
        df = pd.read_sql_query(f"SELECT * FROM {table[0]}", connection)
        df.to_parquet(f"{output_folder}/{table[0]}.parquet", index=False)


    connection.close()

# Convert benchmark and groundtruth databases to Parquets
db_to_parquet(benchmark_db, 'benchmark')
db_to_parquet(groundtruth_db, 'groundtruth')
