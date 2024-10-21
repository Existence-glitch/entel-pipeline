import os
import sys
import time
import logging
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils.ingestion import DataIngestion
from src.utils.preprocessing import DataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingComparison:
    def __init__(self, spark):
        self.spark = spark
        logger.info("Initializing embedding models...")
        self.models = {
            'all-MiniLM-L6-v2': SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        }
        logger.info("Embedding models initialized successfully.")

    def generate_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        logger.info(f"Generating embeddings using {model_name}...")
        start_time = time.time()
        embeddings = self.models[model_name].encode(texts)
        end_time = time.time()
        logger.info(f"Embeddings generated in {end_time - start_time:.2f} seconds.")
        return embeddings

    def evaluate_embeddings(self, embeddings: np.ndarray) -> Dict[str, float]:
        logger.info("Evaluating embeddings...")
        start_time = time.time()
        
        # Compute average cosine similarity
        cos_sim = cosine_similarity(embeddings)
        avg_cos_sim = np.mean(cos_sim)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)  # Assuming 5 clusters, adjust as needed
        cluster_labels = kmeans.fit_predict(embeddings)

        # Compute silhouette score
        silhouette = silhouette_score(embeddings, cluster_labels)

        end_time = time.time()
        logger.info(f"Embeddings evaluated in {end_time - start_time:.2f} seconds.")
        return {
            'avg_cosine_similarity': avg_cos_sim,
            'silhouette_score': silhouette
        }

    def compare_models(self, data: List[Dict]) -> Dict[str, Dict[str, float]]:
        logger.info("Starting model comparison...")
        texts = [item['combined_text'] for item in data]
        results = {}

        for model_name in self.models.keys():
            embeddings = self.generate_embeddings(texts, model_name)
            results[model_name] = self.evaluate_embeddings(embeddings)

        logger.info("Model comparison completed.")
        return results

def main():
    logger.info("Starting embedding comparison process...")
    start_time = time.time()

    # Initialize Spark session
    logger.info("Initializing Spark session...")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("EmbeddingComparison").getOrCreate()
    logger.info("Spark session initialized.")

    # Ingest and preprocess data
    logger.info("Ingesting data...")
    ingestion = DataIngestion(spark)
    raw_data_path = os.path.join(project_root, 'data', 'raw')
    df = ingestion.ingest_and_create_dataframe(raw_data_path)
    logger.info(f"Data ingested. Number of records: {df.count()}")

    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor(spark)
    preprocessed_df = preprocessor.preprocess(df)
    logger.info("Data preprocessing completed.")

    # Convert DataFrame to list of dictionaries
    data = preprocessed_df.select('combined_text').rdd.map(lambda x: x.asDict()).collect()
    logger.info(f"Number of preprocessed records: {len(data)}")

    # Compare embedding models
    comparison = EmbeddingComparison(spark)
    results = comparison.compare_models(data)

    # Print results
    logger.info("Embedding comparison results:")
    for model_name, scores in results.items():
        logger.info(f"\nResults for {model_name}:")
        for metric, value in scores.items():
            logger.info(f"{metric}: {value:.4f}")

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds.")

    spark.stop()

if __name__ == "__main__":
    main()