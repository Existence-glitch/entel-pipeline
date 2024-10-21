from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf, col, lower, regexp_replace, when, concat_ws
from pyspark.sql.types import StringType, ArrayType
import re
import os

class DataPreprocessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def preprocess(self, df: DataFrame) -> DataFrame:
        """
        Preprocess the ingested data for embedding generation.
        
        Args:
        df (DataFrame): Input DataFrame from the ingestion stage
        
        Returns:
        DataFrame: Preprocessed DataFrame
        """
        # Register UDFs
        clean_text_udf = udf(self.clean_text, StringType())
        combine_fields_udf = udf(self.combine_fields, StringType())

        # Clean and normalize text fields
        text_columns = ['fingerprint', 'data', 'cpe']
        for col_name in text_columns:
            df = df.withColumn(col_name, clean_text_udf(col(col_name)))

        # Handle missing values
        df = df.fillna({'fqdn': '', 'cpe': 'unknown'})

        # Convert IP addresses
        df = df.withColumn('ip', regexp_replace('ip', '\.', '_'))

        # Convert 'puerto' to string
        df = df.withColumn('puerto', col('puerto').cast('string'))

        # Combine relevant fields for embedding
        df = df.withColumn('combined_text', combine_fields_udf(
            col('ip'), col('fqdn'), col('puerto'), col('fingerprint'),
            col('cpe'), col('cve'), col('vectores')
        ))

        return df

    @staticmethod
    def clean_text(text):
        """Clean and normalize text."""
        if text is None or text == '':
            return ''
        # Convert to lowercase
        text = text.lower()
        # Remove special characters except underscores, forward slashes, dots, and hyphens
        text = re.sub(r'[^a-z0-9\s_/.-]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    @staticmethod
    def combine_fields(ip, fqdn, puerto, fingerprint, cpe, cve, vectores):
        """Combine relevant fields for embedding generation."""
        combined = f"IP: {ip} "
        if fqdn:
            combined += f"FQDN: {fqdn} "
        combined += f"Port: {puerto} Fingerprint: {fingerprint} "
        if cpe != 'unknown':
            combined += f"CPE: {cpe} "
        if cve:
            combined += f"CVEs: {' '.join(cve)} "
        if vectores:
            combined += f"Vectors: {' '.join(vectores)}"
        return combined.strip()

# Usage example
if __name__ == "__main__":
    spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()
    
    # Ingest data
    from src.utils.ingestion import DataIngestion
    ingestion = DataIngestion(spark)
    raw_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
    input_df = ingestion.ingest_and_create_dataframe(raw_data_path)
    
    # Preprocess data
    preprocessor = DataPreprocessor(spark)
    processed_df = preprocessor.preprocess(input_df)
    
    processed_df.show(5, truncate=False)
    spark.stop()