import os
import sys
import unittest
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, ArrayType, StringType
from pyspark.sql.functions import lit

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils.ingestion import DataIngestion

class TestDataIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("TestEntelPipeline").getOrCreate()
        cls.ingestion = DataIngestion(cls.spark)
        cls.data_path = os.path.join(project_root, 'data', 'raw', 'initial-full-database.db.json')

    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.data_path), f"Data file not found: {self.data_path}")

    def test_ingest_file(self):
        data = self.ingestion.ingest_file(self.data_path)
        self.assertIsNotNone(data, "Data should not be None")
        self.assertIsInstance(data, list, "Data should be a list")
        self.assertTrue(len(data) > 0, "Data should not be empty")

    def test_preprocess_data(self):
        data = self.ingestion.ingest_file(self.data_path)
        preprocessed_data = self.ingestion.preprocess_data(data)
        self.assertIsInstance(preprocessed_data, list, "Preprocessed data should be a list")
        self.assertTrue(len(preprocessed_data) > 0, "Preprocessed data should not be empty")
        
        # Check a few preprocessed items
        for item in preprocessed_data[:5]:  # Check first 5 items
            self.assertIsInstance(item['cve'], list, "CVE should be a list")
            self.assertIsInstance(item['vectores'], list, "Vectores should be a list")
            self.assertTrue(isinstance(item['puerto'], int) or item['puerto'] is None, "Puerto should be an integer or None")

    def test_create_dataframe(self):
        data = self.ingestion.ingest_file(self.data_path)
        df = self.ingestion.create_dataframe(data)
        self.assertIsNotNone(df, "DataFrame should not be None")
        self.assertTrue(df.count() > 0, "DataFrame should not be empty")

        # Check if all expected columns are present
        expected_columns = ['did', 'ip', 'fqdn', 'puerto', 'fingerprint', 'data', 'cpe', 'cve', 'vectores']
        self.assertListEqual(df.columns, expected_columns, "DataFrame columns do not match expected columns")

        # Check if 'puerto' column is of IntegerType
        self.assertIsInstance(df.schema['puerto'].dataType, IntegerType, "'puerto' column should be IntegerType")

        # Check if 'cve' column is of ArrayType(StringType)
        self.assertIsInstance(df.schema['cve'].dataType, ArrayType, "'cve' column should be ArrayType")
        self.assertIsInstance(df.schema['cve'].dataType.elementType, StringType, "'cve' column elements should be StringType")

    def test_ingest_and_create_dataframe(self):
        df = self.ingestion.ingest_and_create_dataframe(self.data_path)
        self.assertIsNotNone(df, "DataFrame should not be None")
        self.assertTrue(df.count() > 0, "DataFrame should not be empty")

        # Print some basic statistics about the dataframe
        print(f"Number of records: {df.count()}")
        print("Sample data:")
        df.show(5, truncate=False)

        # Check data types of columns
        print("DataFrame schema:")
        df.printSchema()

        # Check for null values in 'puerto' column
        null_puerto_count = df.filter(df.puerto.isNull()).count()
        print(f"Number of null values in 'puerto' column: {null_puerto_count}")

        # Check 'cve' column
        empty_cve_count = df.filter(df.cve.isNull() | (df.cve == lit([])) | (df.cve == lit(['']))).count()
        print(f"Number of empty or null values in 'cve' column: {empty_cve_count}")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

if __name__ == '__main__':
    unittest.main()