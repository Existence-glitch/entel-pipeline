import os
import json
import ast
from typing import List, Dict, Union
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from pyspark.sql.functions import col, when, lit

class DataIngestion:
    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder.appName("EntelPipeline").getOrCreate()

    def _define_schema(self) -> StructType:
        return StructType([
            StructField("did", StringType(), nullable=False),
            StructField("ip", StringType(), nullable=True),
            StructField("fqdn", StringType(), nullable=True),
            StructField("puerto", IntegerType(), nullable=True),
            StructField("fingerprint", StringType(), nullable=True),
            StructField("data", StringType(), nullable=True),
            StructField("cpe", StringType(), nullable=True),
            StructField("cve", ArrayType(StringType()), nullable=True),
            StructField("vectores", ArrayType(StringType()), nullable=True)
        ])

    def read_json_file(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r') as file:
            return json.load(file)

    def ingest_file(self, file_path: str) -> Union[List[Dict], None]:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        data = self.read_json_file(file_path)
        return data

    def ingest_directory(self, directory_path: str) -> List[Dict]:
        all_data = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                file_path = os.path.join(directory_path, filename)
                file_data = self.ingest_file(file_path)
                if file_data:
                    all_data.extend(file_data)
        return all_data

    def preprocess_data(self, data: List[Dict]) -> List[Dict]:
        def parse_cve(cve):
            if isinstance(cve, list):
                return cve
            elif isinstance(cve, str):
                try:
                    return ast.literal_eval(cve)
                except:
                    return []
            else:
                return []

        def parse_puerto(puerto):
            try:
                return int(puerto)
            except (ValueError, TypeError):
                return None

        preprocessed_data = []
        for item in data:
            preprocessed_item = item.copy()
            preprocessed_item['cve'] = parse_cve(item.get('cve', []))
            preprocessed_item['vectores'] = item.get('vectores', [])
            preprocessed_item['puerto'] = parse_puerto(item.get('puerto'))
            preprocessed_data.append(preprocessed_item)

        return preprocessed_data

    def create_dataframe(self, data: List[Dict]) -> 'pyspark.sql.DataFrame':
        schema = self._define_schema()
        preprocessed_data = self.preprocess_data(data)
        df = self.spark.createDataFrame(preprocessed_data, schema)
        return df

    def ingest_and_create_dataframe(self, path: str) -> 'pyspark.sql.DataFrame':
        if os.path.isfile(path):
            data = self.ingest_file(path)
        elif os.path.isdir(path):
            data = self.ingest_directory(path)
        else:
            raise ValueError(f"Invalid path: {path}")
        
        return self.create_dataframe(data)

# Usage example
if __name__ == "__main__":
    ingestion = DataIngestion()
    
    # Adjust this path to point to your data/raw directory
    raw_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
    
    df = ingestion.ingest_and_create_dataframe(raw_data_path)
    print(f"Ingested {df.count()} records")
    df.show(5, truncate=False)