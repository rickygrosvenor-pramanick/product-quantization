"""
Data Processing Pipeline for Product Quantization

This module processes the sentiment data.csv and creates embeddings using Cohere
for training the Product Quantizer. It also provides a simple retrieval system
without requiring a traditional vector database.

Author: Ricky Pramanick
Date: July 2025
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our modules
from pq import ProductQuantizer, brute_force_search

try:
    from embeddings import EmbeddingGenerator, save_embeddings, load_embeddings
except ImportError:
    EmbeddingGenerator = None
    save_embeddings = None
    load_embeddings = None


class DataProcessor:
    """
    Processes the sentiment dataset and prepares it for PQ training.
    """
    
    def __init__(self, csv_path: str = "data.csv"):
        """
        Initialize data processor.
        
        Args:
            csv_path (str): Path to the CSV file containing sentences and sentiments
        """
        self.csv_path = csv_path
        self.df = None
        self.embeddings = None
        self.texts = None
        self.sentiments = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the CSV data.
        
        Returns:
            pd.DataFrame: Processed dataframe
        """
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nSentiment distribution:")
        print(self.df['Sentiment'].value_counts())
        
        # Clean the text data
        self.df['Sentence'] = self.df['Sentence'].astype(str)
        self.df['Sentence'] = self.df['Sentence'].str.strip()
        
        # Remove any empty sentences
        self.df = self.df[self.df['Sentence'] != '']
        
        print(f"After cleaning: {len(self.df)} sentences")
        
        self.texts = self.df['Sentence'].tolist()
        self.sentiments = self.df['Sentiment'].tolist()
        
        return self.df
    
    def create_embeddings(self, model: str = "embed-english-light-v3.0", 
                         batch_size: int = 96) -> np.ndarray:
        """
        Create embeddings for all sentences using Cohere.
        
        Args:
            model (str): Cohere model to use
            batch_size (int): Batch size for API calls
            
        Returns:
            np.ndarray: Embeddings array of shape (N, embedding_dim)
        """
        print(f"\nCreating embeddings using {model}...")
        
        # Initialize Cohere embedder
        embedder = EmbeddingGenerator(model=model)
        
        # Generate embeddings
        self.embeddings = embedder.embed_texts(
            self.texts, 
            batch_size=batch_size, 
            show_progress=True
        )
        
        print(f"Created embeddings shape: {self.embeddings.shape}")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
        
        return self.embeddings
    
    def save_processed_data(self, base_path: str = "processed_data"):
        """
        Save processed embeddings and metadata.
        
        Args:
            base_path (str): Base path for saving files
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Run create_embeddings() first.")
        
        # Save embeddings and texts
        save_embeddings(self.embeddings, self.texts, base_path)
        
        # Save metadata
        metadata = {
            'sentiments': self.sentiments,
            'embedding_dim': self.embeddings.shape[1],
            'num_samples': len(self.texts),
            'sentiment_counts': pd.Series(self.sentiments).value_counts().to_dict()
        }
        
        with open(f"{base_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved processed data to {base_path}.*")
    
    def load_processed_data(self, base_path: str = "processed_data"):
        """
        Load previously processed data.
        
        Args:
            base_path (str): Base path for loading files
        """
        # Load embeddings and texts
        self.embeddings, self.texts = load_embeddings(base_path)
        
        # Load metadata
        with open(f"{base_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.sentiments = metadata['sentiments']
        
        print(f"Loaded {len(self.texts)} samples with {self.embeddings.shape[1]}D embeddings")


class SimpleRetrievalSystem:
    """
    Retrieval system using Product Quantization without a vector database.

    Supports two search paths:
      1. **PQ-only** (default): asymmetric distance over all PQ codes — no
         external index needed, works well up to ~100k documents.
      2. **Re-ranked**: shortlist via PQ, then re-rank with exact L2 distances
         against the stored original embeddings for higher precision.
    """

    def __init__(self, M: int = 8, K: int = 256):
        self.M = M
        self.K = K
        self.pq = ProductQuantizer(M=M, K=K, verbose=True)

        self.codes = None
        self.texts = None
        self.metadata = None
        self.original_embeddings = None

    def train_quantizer(self, embeddings: np.ndarray) -> None:
        print(f"\nTraining Product Quantizer with M={self.M}, K={self.K}...")
        print(f"Input embeddings shape: {embeddings.shape}")

        self.pq.fit(embeddings)

        print("Product Quantizer training completed!")

        original_size = embeddings.nbytes
        code_size = embeddings.shape[0] * self.M * (1 if self.K <= 256 else 2)
        codebook_size = self.M * self.K * (embeddings.shape[1] // self.M) * 4
        total_pq_size = code_size + codebook_size
        compression_ratio = original_size / total_pq_size

        print(f"\nCompression Analysis:")
        print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        print(f"PQ codes size: {code_size / 1024:.2f} KB")
        print(f"Codebooks size: {codebook_size / 1024:.2f} KB")
        print(f"Total PQ size: {total_pq_size / 1024:.2f} KB")
        print(f"Compression ratio: {compression_ratio:.1f}x")

    def index_documents(self, embeddings: np.ndarray, texts: List[str],
                       metadata: Optional[List] = None) -> None:
        if not self.pq.is_trained:
            raise RuntimeError("Quantizer must be trained before indexing")

        print(f"\nIndexing {len(texts)} documents...")

        self.codes = self.pq.encode(embeddings)
        self.texts = texts
        self.metadata = metadata
        self.original_embeddings = embeddings

        # Pre-compute the distance table lookup array for fast search
        self._precompute_lookup()

        print(f"Indexed documents with PQ codes shape: {self.codes.shape}")

        memory_usage = self.get_memory_usage()
        print(f"\nMemory Usage:")
        for key, value in memory_usage.items():
            print(f"{key}: {value / 1024:.2f} KB")

    def _precompute_lookup(self) -> None:
        """Build a flat int array of codes for cache-friendly distance lookup."""
        if self.codes is not None:
            self._codes_int = self.codes.astype(np.intp)

    def search(self, query_embedding: np.ndarray, k: int = 10,
               rerank: bool = False, rerank_factor: int = 4,
               sentiment_filter: Optional[str] = None) -> Tuple[List[str], List[float], List]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector of shape (D,)
            k: Number of results to return
            rerank: If True, shortlist with PQ then re-rank with exact L2
            rerank_factor: How many PQ candidates to fetch before re-ranking
            sentiment_filter: If set, only return documents with this sentiment
        """
        if self.codes is None:
            raise RuntimeError("No documents indexed")

        pq_distances = self.pq.asymmetric_distance(query_embedding, self.codes)

        if sentiment_filter and self.metadata:
            mask = np.array([m == sentiment_filter for m in self.metadata])
            pq_distances[~mask] = np.inf

        if rerank and self.original_embeddings is not None:
            n_candidates = min(k * rerank_factor, len(self.texts))
            candidate_indices = np.argpartition(pq_distances, n_candidates)[:n_candidates]

            exact_distances = np.linalg.norm(
                self.original_embeddings[candidate_indices] - query_embedding[np.newaxis, :],
                axis=1
            )
            top_k_local = np.argsort(exact_distances)[:k]
            top_k_indices = candidate_indices[top_k_local]
            result_distances = exact_distances[top_k_local].tolist()
        else:
            top_k_indices = np.argsort(pq_distances)[:k]
            result_distances = pq_distances[top_k_indices].tolist()

        results_texts = [self.texts[i] for i in top_k_indices]
        results_metadata = [self.metadata[i] if self.metadata else None for i in top_k_indices]

        return results_texts, result_distances, results_metadata

    def search_by_text(self, query_text: str, embedder: EmbeddingGenerator,
                      k: int = 10, rerank: bool = False) -> Tuple[List[str], List[float], List]:
        query_embedding = embedder.embed_queries([query_text])[0]
        return self.search(query_embedding, k, rerank=rerank)

    def batch_search(self, query_embeddings: np.ndarray, k: int = 10,
                     rerank: bool = False) -> List[Tuple[List[str], List[float], List]]:
        """Run search for multiple query vectors at once."""
        return [self.search(q, k=k, rerank=rerank) for q in query_embeddings]

    def evaluate_recall(self, query_embeddings: np.ndarray, k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        if self.original_embeddings is None:
            raise RuntimeError("Original embeddings not available for evaluation")

        print(f"\nEvaluating recall with {len(query_embeddings)} queries...")

        max_k = max(k_values)

        true_neighbors = []
        for query in tqdm(query_embeddings, desc="Computing ground truth"):
            _, indices = brute_force_search(self.original_embeddings, query.reshape(1, -1), k=max_k)
            true_neighbors.append(indices[0])
        true_neighbors = np.array(true_neighbors)

        pq_neighbors = []
        for query in tqdm(query_embeddings, desc="PQ search"):
            distances = self.pq.asymmetric_distance(query, self.codes)
            indices = np.argsort(distances)[:max_k]
            pq_neighbors.append(indices)
        pq_neighbors = np.array(pq_neighbors)

        recall_results = {}
        for k in k_values:
            recall_sum = 0
            for i in range(len(query_embeddings)):
                true_set = set(true_neighbors[i][:k])
                pred_set = set(pq_neighbors[i][:k])
                recall_sum += len(true_set & pred_set) / len(true_set)

            recall_results[k] = recall_sum / len(query_embeddings)
            print(f"Recall@{k}: {recall_results[k]:.3f}")

        return recall_results

    def get_memory_usage(self) -> Dict[str, float]:
        if self.codes is None:
            return {}

        usage = {}
        usage['PQ codes'] = self.codes.nbytes
        usage['Text storage'] = sum(len(text.encode('utf-8')) for text in self.texts)
        usage['Codebooks'] = self.pq.codebooks.nbytes if self.pq.codebooks is not None else 0

        if self.metadata:
            usage['Metadata'] = sum(len(str(meta).encode('utf-8')) for meta in self.metadata)

        return usage

    def save_index(self, filepath: str) -> None:
        data = {
            'pq': self.pq,
            'codes': self.codes,
            'texts': self.texts,
            'metadata': self.metadata,
            'M': self.M,
            'K': self.K
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved retrieval system to {filepath}")

    def load_index(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.pq = data['pq']
        self.codes = data['codes']
        self.texts = data['texts']
        self.metadata = data['metadata']
        self.M = data['M']
        self.K = data['K']

        if self.codes is not None:
            self._precompute_lookup()

        print(f"Loaded retrieval system from {filepath}")


def main():
    """
    Main pipeline for processing data and creating a retrieval system.
    """
    print("=== Product Quantization Data Processing Pipeline ===")
    
    # Step 1: Load and process data
    processor = DataProcessor()
    processor.load_data()
    
    # Check if embeddings already exist
    if os.path.exists("processed_data.npz"):
        print("\nFound existing embeddings, loading...")
        processor.load_processed_data()
    else:
        print("\nCreating new embeddings...")
        try:
            processor.create_embeddings()
            processor.save_processed_data()
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            print("Falling back to random embeddings for demo...")
            
            # Generate random embeddings as fallback
            from pq import generate_random_embeddings
            processor.embeddings = generate_random_embeddings(
                len(processor.texts), 384, random_state=42
            )
            processor.save_processed_data()
            print("Generated random embeddings for demonstration")
    
    # Step 2: Create retrieval system
    print("\n=== Setting up Retrieval System ===")
    
    # Split data into train/test
    n_samples = len(processor.texts)
    n_train = int(0.8 * n_samples)
    
    train_embeddings = processor.embeddings[:n_train]
    test_embeddings = processor.embeddings[n_train:]
    test_texts = processor.texts[n_train:]
    test_sentiments = processor.sentiments[n_train:]
    
    # Initialize and train retrieval system
    retrieval_system = SimpleRetrievalSystem(M=8, K=256)
    retrieval_system.train_quantizer(train_embeddings)
    
    # Index all documents (including training ones for complete search)
    retrieval_system.index_documents(
        processor.embeddings, 
        processor.texts, 
        processor.sentiments
    )
    
    # Step 3: Evaluation
    print("\n=== Evaluation ===")
    
    # Evaluate recall with a subset of test queries
    n_eval_queries = min(50, len(test_embeddings))
    eval_embeddings = test_embeddings[:n_eval_queries]
    
    recall_results = retrieval_system.evaluate_recall(eval_embeddings)
    
    # Step 4: Demo search
    print("\n=== Demo Search ===")
    
    # Try to search with embedder (if available)
    try:
        embedder = EmbeddingGenerator()
        
        print("\n--- Interactive Search Demo ---")
        print("Enter your search queries (press Enter with empty query to finish):")
        print("Results will be saved to 'search_results.txt'")
        
        # Open file for writing search results
        with open("search_results.txt", "w", encoding="utf-8") as f:
            f.write("Search Results\n")
            f.write("=" * 50 + "\n\n")
            
            while True:
                query = input("\nEnter query: ").strip()
                if not query:
                    break
                    
                print(f"\nSearching for: '{query}'")
                results, distances, sentiments = retrieval_system.search_by_text(
                    query, embedder, k=3
                )
                
                # Write to file
                f.write(f"Query: {query}\n")
                f.write("-" * 30 + "\n")
                
                print("Top 3 results:")
                for i, (text, distance, sentiment) in enumerate(zip(results, distances, sentiments)):
                    # Write to file (text and distance only)
                    f.write(f"{distance:.6f}\t{text}\n")
                    
                    # Print to console (with sentiment for display)
                    print(f"{i+1}. [{sentiment}] (dist: {distance:.3f})")
                    print(f"   {text[:100]}...")
                
                f.write("\n")
                f.flush()  # Ensure data is written immediately
        
        print("\nDemo search completed!")
        print("Results saved to 'search_results.txt'")
    
    except Exception as e:
        print(f"Demo search failed: {e}")
        print("To enable text search, set COHERE_API_KEY environment variable")
    
    # Step 5: Save the system
    retrieval_system.save_index("sentiment_retrieval_system.pkl")
    
    print("\n=== Pipeline Complete ===")
    print("Your retrieval system is ready!")


if __name__ == "__main__":
    main()
