# src/main.py

import os
from pyspark.sql import SparkSession
from src.utils.ingestion import DataIngestion
from src.utils.preprocessing import DataPreprocessor
from src.utils.embedding import EmbeddingGenerator

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("EntelReconRAGPipeline").getOrCreate()

    # Initialize pipeline components
    ingestion = DataIngestion(spark)
    preprocessor = DataPreprocessor(spark)
    embedding_generator = EmbeddingGenerator()

    # Set up paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, 'data', 'raw')
    processed_data_path = os.path.join(project_root, 'data', 'processed')
    embeddings_path = os.path.join(project_root, 'data', 'embeddings')

    # Ensure output directories exist
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(embeddings_path, exist_ok=True)

    # Execute pipeline
    print("Ingesting data...")
    df = ingestion.ingest_and_create_dataframe(raw_data_path)
    
    print("Preprocessing data...")
    df_preprocessed = preprocessor.preprocess(df)
    
    print("Generating embeddings...")
    df_with_embeddings = embedding_generator.generate_embeddings(df_preprocessed)

    # Save results
    print("Saving processed data and embeddings...")
    df_preprocessed.write.parquet(os.path.join(processed_data_path, 'processed_data.parquet'))
    df_with_embeddings.write.parquet(os.path.join(embeddings_path, 'data_with_embeddings.parquet'))

    print("Pipeline execution completed.")
    spark.stop()

if __name__ == "__main__":
    main()