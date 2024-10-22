import json
import faiss
import numpy as np
import os
from openai import OpenAI
import logging
import time
from datetime import datetime

class SecurityDataChat:
    def __init__(self, data_file, index_file='faiss_index.idx', embedding_file='embeddings.npy', embedding_model="vonjack/bge-m3-gguf", log_file='LLM_model.log'):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.embedding_model = embedding_model
        self.data_file = data_file
        self.index_file = index_file
        self.embedding_file = embedding_file
        self.data = self.load_data()
        self.ip_to_indices = self.create_ip_index()
        self.embeddings, self.index = self.load_or_create_index()

        self.logger.info("SecurityDataChat initialized successfully.")

    def load_data(self):
        with open(self.data_file, 'r') as f:
            return json.load(f)

    def create_ip_index(self):
        ip_to_indices = {}
        for i, item in enumerate(self.data):
            ip = item['ip']
            if ip in ip_to_indices:
                ip_to_indices[ip].append(i)
            else:
                ip_to_indices[ip] = [i]
        return ip_to_indices

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.embedding_model).data[0].embedding

    def create_input_text(self, item):
        return f"IP: {item['ip']}\nFQDN: {item['fqdn']}\nPort: {item['puerto']}\nFingerprint: {item['fingerprint']}\nData: {item['data']}\nCPE: {item['cpe']}\nCVE: {', '.join(item['cve'])}\nVectores: {', '.join(item['vectores'])}"

    def create_index(self):
        self.logger.info("Creating new index and embeddings...")
        texts = [self.create_input_text(item) for item in self.data]
        embeddings = np.array([self.get_embedding(text) for text in texts])
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        return embeddings, index

    def save_index_and_embeddings(self, index, embeddings):
        self.logger.info(f"Saving index to {self.index_file} and embeddings to {self.embedding_file}...")
        faiss.write_index(index, self.index_file)
        np.save(self.embedding_file, embeddings)

    def load_index_and_embeddings(self):
        self.logger.info(f"Loading index from {self.index_file} and embeddings from {self.embedding_file}...")
        index = faiss.read_index(self.index_file)
        embeddings = np.load(self.embedding_file)
        return embeddings, index

    def load_or_create_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.embedding_file):
            return self.load_index_and_embeddings()
        else:
            embeddings, index = self.create_index()
            self.save_index_and_embeddings(index, embeddings)
            return embeddings, index

    def retrieve_relevant_items(self, query, k=5):
        # First, try to match by IP
        if query in self.ip_to_indices:
            relevant_indices = self.ip_to_indices[query]
            return [self.data[i] for i in relevant_indices]
        
        # If no exact IP match, use similarity search
        query_embedding = np.array([self.get_embedding(query)])
        distances, indices = self.index.search(query_embedding, k)
        return [self.data[i] for i in indices[0]]

    def generate_response(self, query, relevant_items):
        prompt = f"""Given the following query:
{query}

And these relevant data points:
{json.dumps(relevant_items, indent=2)}

First check that the query IP matches the one in the relevant data points, if it doesn't tell me so I can know that the IP is not in the database, else if it does match then
Provide the most relevant data found for the queried IP, hostname or fqdn 
for a pentester to know what are the vulnerabilities found and recommend ways to exploit them."""

        start_time = time.time()
        response = self.client.chat.completions.create(
            model="bartowski/Ministral-8B-Instruct-2410-HF-GGUF-TEST",  # Replace with your preferred model
            messages=[
                {"role": "system", "content": "You are an expert in cybersecurity and you identify and retrieve information about specific IP's, hostnames, open ports and its associated attack vectors."},
                {"role": "user", "content": prompt}
            ]
        )
        end_time = time.time()
        response_time = end_time - start_time

        self.logger.info(f"Query: {query}")
        self.logger.info(f"Response time: {response_time:.2f} seconds")
        self.logger.info(f"Response: {response.choices[0].message.content}")

        return response.choices[0].message.content, response_time

    def chat_query(self, query):
        relevant_items = self.retrieve_relevant_items(query)
        return self.generate_response(query, relevant_items)

# Usage
if __name__ == "__main__":
    raw_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
    path = raw_data_path + '/sample-dataset.json'
    chat = SecurityDataChat(path)
    
    print("Entel-pipeline initialized. You can now start querying.")
    print("Enter an IP, IP/port pair, or FQDN. Type 'quit' to exit.")
    
    while True:
        user_query = input("\nQuery: ")
        if user_query.lower() == 'quit':
            break
        
        response, response_time = chat.chat_query(user_query)
        print("\nResponse:")
        print(response)
        print(f"\nResponse time: {response_time:.2f} seconds")
        print("\n" + "-"*50)

    print("Thank you for using Security Data Chat. Goodbye!")