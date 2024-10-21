# src/data/embedding.py

import os
from openai import OpenAI
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd

class EmbeddingGenerator:
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def get_embedding(self, text: str, model: str = "model-identifier") -> list:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def generate_embeddings(self, df, text_column: str = 'combined_text', embedding_column: str = 'embedding'):
        # Define the Pandas UDF
        @pandas_udf(ArrayType(FloatType()))
        def embedding_udf(texts: pd.Series) -> pd.Series:
            return texts.apply(lambda x: self.get_embedding(x))

        # Apply the Pandas UDF to create the embedding column
        return df.withColumn(embedding_column, embedding_udf(df[text_column]))

# Usage example
if __name__ == "__main__":
    spark = SparkSession.builder.appName("EmbeddingGeneration").getOrCreate()
    
    # Assuming you have already run the ingestion and preprocessing steps
    from src.utils.ingestion import DataIngestion
    from src.utils.preprocessing import DataPreprocessor
    
    ingestion = DataIngestion(spark)
    preprocessor = DataPreprocessor(spark)
    
    # Ingest and preprocess data
    raw_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
    df = ingestion.ingest_and_create_dataframe(raw_data_path)
    df_preprocessed = preprocessor.preprocess(df)
    
    # Generate embeddings
    embedding_generator = EmbeddingGenerator()
    df_with_embeddings = embedding_generator.generate_embeddings(df_preprocessed)
    
    # Show sample results
    df_with_embeddings.select('did', 'combined_text', 'embedding').show(5, truncate=True)
    
    spark.stop()