"""
Embedding Generation Module for Product Quantization

This module provides utilities for generating embeddings using Cohere's API
with small dimensions suitable for PQ experimentation and testing.

Author: Ricky Pramanick
Date: July 2025
"""

import numpy as np
import cohere
import os
import random
from typing import List, Optional
from tqdm import tqdm


class EmbeddingGenerator:
    """
    Embedding generator using Cohere API for creating vector representations.
    
    Uses Cohere's embed-english-light-v3.0 model which produces 384-dimensional
    embeddings - a good size for PQ experimentation (small enough to be fast,
    large enough to demonstrate compression benefits).
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "embed-english-light-v3.0"):
        """
        Initialize embedding generator.
        
        Args:
            api_key (Optional[str]): Cohere API key. If None, reads from COHERE_API_KEY env var
            model (str): Cohere embedding model to use
                       - "embed-english-light-v3.0": 384 dimensions, fast
                       - "embed-english-v3.0": 1024 dimensions, more accurate
                       - "embed-multilingual-light-v3.0": 384 dimensions, multilingual
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required. Set COHERE_API_KEY env var or pass api_key")
        
        self.model = model
        self.client = cohere.Client(self.api_key)
        
        # Model dimensions for memory calculations
        self.model_dims = {
            "embed-english-light-v3.0": 384,
            "embed-english-v3.0": 1024,
            "embed-multilingual-light-v3.0": 384,
            "embed-multilingual-v3.0": 1024
        }
        self.embedding_dim = self.model_dims.get(model, 384)
        
        print(f"Initialized Cohere embeddings with model: {model}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 96, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to embed
            batch_size (int): Batch size for API calls (Cohere limit is 96)
            show_progress (bool): Whether to show progress bar
            
        Returns:
            np.ndarray: Embeddings array of shape (len(texts), embedding_dim)
        """
        embeddings = []
        
        # Process in batches to respect API limits
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        if show_progress:
            batches = tqdm(batches, desc="Generating embeddings")
        
        for batch in batches:
            try:
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_document"  # For indexing/search use case
                )
                
                batch_embeddings = np.array(response.embeddings, dtype=np.float32)
                embeddings.append(batch_embeddings)
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Add zero embeddings for failed batch
                batch_embeddings = np.zeros((len(batch), self.embedding_dim), dtype=np.float32)
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def embed_queries(self, queries: List[str], batch_size: int = 96, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for query texts (optimized for search).
        
        Args:
            queries (List[str]): List of query strings to embed
            batch_size (int): Batch size for API calls
            show_progress (bool): Whether to show progress bar
            
        Returns:
            np.ndarray: Query embeddings array of shape (len(queries), embedding_dim)
        """
        embeddings = []
        
        batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        
        if show_progress:
            batches = tqdm(batches, desc="Generating query embeddings")
        
        for batch in batches:
            try:
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_query"  # Optimized for query use case
                )
                
                batch_embeddings = np.array(response.embeddings, dtype=np.float32)
                embeddings.append(batch_embeddings)
                
            except Exception as e:
                print(f"Error processing query batch: {e}")
                # Add zero embeddings for failed batch
                batch_embeddings = np.zeros((len(batch), self.embedding_dim), dtype=np.float32)
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)


def create_sample_documents(n_docs: int = 1000, seed: int = 42) -> List[str]:
    """
    Create sample documents for embedding and PQ testing.
    
    Generates diverse text samples that would be realistic for a
    document search or recommendation system.
    
    Args:
        n_docs (int): Number of sample documents to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        List[str]: List of sample document texts
    """
    random.seed(seed)
    
    # Sample document templates for different domains
    templates = [
        "Machine learning techniques for {topic} have shown significant improvements in {metric}.",
        "The latest research in {field} demonstrates that {method} outperforms traditional approaches.",
        "A comprehensive study of {subject} reveals new insights about {aspect} and {application}.",
        "Recent developments in {technology} have revolutionized how we approach {problem}.",
        "Analysis of {dataset} using {technique} provides evidence for {conclusion}.",
        "The impact of {factor} on {outcome} has been studied extensively in {domain}.",
        "Novel approaches to {challenge} leverage {tool} to achieve better {performance}.",
        "Experimental results show that {approach} significantly improves {measure} in {context}.",
        "Deep learning models trained on {data} demonstrate superior {quality} compared to baseline methods.",
        "Optimization algorithms for {task} can reduce computational {cost} while maintaining {accuracy}.",
        "Real-world applications of {technology} in {industry} have yielded promising {results}.",
        "Comparative analysis between {method1} and {method2} reveals trade-offs in {metric1} vs {metric2}."
    ]
    
    # Sample content for template filling
    topics = ["image recognition", "natural language processing", "recommendation systems", 
              "time series forecasting", "anomaly detection", "clustering", "classification",
              "sentiment analysis", "object detection", "speech recognition"]
    
    fields = ["computer vision", "artificial intelligence", "data science", "deep learning",
              "information retrieval", "computational linguistics", "pattern recognition",
              "machine learning", "neural networks", "statistical modeling"]
    
    methods = ["neural networks", "transformer models", "ensemble methods", "gradient boosting",
               "support vector machines", "random forests", "convolutional networks",
               "recurrent networks", "attention mechanisms", "graph neural networks"]
    
    technologies = ["cloud computing", "edge computing", "distributed systems", "blockchain",
                   "quantum computing", "augmented reality", "Internet of Things",
                   "federated learning", "autonomous systems", "robotics"]
    
    subjects = ["data preprocessing", "feature engineering", "model interpretability",
               "transfer learning", "few-shot learning", "multi-task learning",
               "reinforcement learning", "unsupervised learning", "semi-supervised learning"]
    
    # Generic terms for filling placeholders
    metrics = ["accuracy", "precision", "recall", "F1-score", "AUC", "BLEU score", "perplexity"]
    aspects = ["robustness", "scalability", "efficiency", "interpretability", "fairness"]
    applications = ["healthcare", "finance", "education", "entertainment", "transportation"]
    problems = ["data scarcity", "computational complexity", "noise handling", "bias mitigation"]
    
    all_terms = {
        "{topic}": topics,
        "{field}": fields,
        "{method}": methods,
        "{method1}": methods,
        "{method2}": methods,
        "{technology}": technologies,
        "{subject}": subjects,
        "{technique}": methods,
        "{tool}": technologies,
        "{approach}": methods,
        "{metric}": metrics,
        "{metric1}": metrics,
        "{metric2}": metrics,
        "{aspect}": aspects,
        "{application}": applications,
        "{problem}": problems,
        "{outcome}": metrics,
        "{challenge}": problems,
        "{performance}": metrics,
        "{measure}": metrics,
        "{context}": applications,
        "{conclusion}": aspects,
        "{dataset}": ["ImageNet", "BERT corpus", "Wikipedia", "Common Crawl", "scientific papers"],
        "{factor}": ["data quality", "model size", "training time", "hyperparameters"],
        "{domain}": applications,
        "{data}": ["text data", "image data", "time series", "graph data", "multimodal data"],
        "{quality}": metrics,
        "{task}": topics,
        "{cost}": ["memory usage", "training time", "inference time", "computational cost"],
        "{accuracy}": metrics,
        "{industry}": ["healthcare", "automotive", "finance", "retail", "manufacturing"],
        "{results}": ["outcomes", "improvements", "breakthroughs", "innovations"]
    }
    
    documents = []
    
    for i in range(n_docs):
        template = random.choice(templates)
        
        # Fill all placeholders in the template
        filled_template = template
        for placeholder, options in all_terms.items():
            if placeholder in filled_template:
                replacement = random.choice(options)
                filled_template = filled_template.replace(placeholder, replacement, 1)
        
        documents.append(filled_template)
    
    return documents


def create_sample_queries(n_queries: int = 100, seed: int = 42) -> List[str]:
    """
    Create sample search queries for testing.
    
    Args:
        n_queries (int): Number of queries to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        List[str]: List of sample query strings
    """
    random.seed(seed)
    
    query_templates = [
        "How to improve {metric} in {task}?",
        "Best {method} for {application}",
        "{technology} applications in {domain}",
        "Comparison of {method1} vs {method2}",
        "Latest research in {field}",
        "{problem} solutions using {technology}",
        "Tutorial on {technique}",
        "State of the art {task} methods",
        "Optimization techniques for {challenge}",
        "Real-world examples of {application}"
    ]
    
    # Simplified terms for queries
    tasks = ["classification", "clustering", "prediction", "detection", "recognition"]
    methods = ["neural networks", "SVM", "random forest", "transformers", "CNN"]
    applications = ["healthcare", "finance", "autonomous driving", "NLP", "computer vision"]
    domains = ["machine learning", "AI", "data science", "deep learning"]
    technologies = ["deep learning", "reinforcement learning", "transfer learning"]
    problems = ["overfitting", "data imbalance", "computational cost", "interpretability"]
    techniques = ["feature selection", "hyperparameter tuning", "model compression"]
    challenges = ["large datasets", "limited data", "real-time processing"]
    
    query_terms = {
        "{metric}": ["accuracy", "speed", "efficiency", "performance"],
        "{task}": tasks,
        "{application}": applications,
        "{method}": methods,
        "{method1}": methods,
        "{method2}": methods,
        "{technology}": technologies,
        "{domain}": domains,
        "{field}": domains,
        "{problem}": problems,
        "{technique}": techniques,
        "{challenge}": challenges
    }
    
    queries = []
    
    for i in range(n_queries):
        template = random.choice(query_templates)
        
        # Fill placeholders
        filled_query = template
        for placeholder, options in query_terms.items():
            if placeholder in filled_query:
                replacement = random.choice(options)
                filled_query = filled_query.replace(placeholder, replacement, 1)
        
        queries.append(filled_query)
    
    return queries


def generate_random_embeddings(n_vectors: int, dimension: int = 384, 
                             random_state: int = 42) -> np.ndarray:
    """
    Generate random embeddings for testing (fallback when API not available).
    
    Creates normalized random vectors that simulate real embeddings
    for benchmarking and testing purposes.
    
    Args:
        n_vectors (int): Number of vectors to generate
        dimension (int): Vector dimension (default 384 to match Cohere light model)
        random_state (int): Random seed
        
    Returns:
        np.ndarray: Random embeddings of shape (n_vectors, dimension)
    """
    np.random.seed(random_state)
    
    # Generate random vectors
    embeddings = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    # Normalize to unit length (common for embedding models)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero
    
    return embeddings


def save_embeddings(embeddings: np.ndarray, texts: List[str], filepath: str) -> None:
    """
    Save embeddings and their corresponding texts to disk.
    
    Args:
        embeddings (np.ndarray): Embedding vectors
        texts (List[str]): Corresponding text data
        filepath (str): Path to save the data (without extension)
    """
    np.savez_compressed(
        f"{filepath}.npz",
        embeddings=embeddings,
        texts=np.array(texts, dtype=object)
    )
    print(f"Saved {len(embeddings)} embeddings to {filepath}.npz")


def load_embeddings(filepath: str) -> tuple[np.ndarray, List[str]]:
    """
    Load embeddings and texts from disk.
    
    Args:
        filepath (str): Path to load from (without extension)
        
    Returns:
        tuple: (embeddings, texts)
    """
    data = np.load(f"{filepath}.npz", allow_pickle=True)
    embeddings = data['embeddings']
    texts = data['texts'].tolist()
    
    print(f"Loaded {len(embeddings)} embeddings from {filepath}.npz")
    return embeddings, texts


if __name__ == "__main__":
    """
    Example usage of the embedding module.
    """
    print("=== Embedding Module Demo ===")
    
    # Generate sample data
    print("\n1. Generating sample documents...")
    documents = create_sample_documents(n_docs=50)
    queries = create_sample_queries(n_queries=10)
    
    print(f"Created {len(documents)} documents and {len(queries)} queries")
    print(f"Sample document: {documents[0]}")
    print(f"Sample query: {queries[0]}")
    
    # Try to use Cohere API (falls back to random if not available)
    try:
        print("\n2. Generating embeddings with Cohere...")
        embedder = EmbeddingGenerator()
        
        doc_embeddings = embedder.embed_texts(documents[:10], show_progress=True)
        query_embeddings = embedder.embed_queries(queries[:5], show_progress=True)
        
        print(f"Document embeddings shape: {doc_embeddings.shape}")
        print(f"Query embeddings shape: {query_embeddings.shape}")
        
    except Exception as e:
        print(f"\n2. Cohere API not available ({e}), using random embeddings...")
        doc_embeddings = generate_random_embeddings(10, dimension=384)
        query_embeddings = generate_random_embeddings(5, dimension=384)
        
        print(f"Random document embeddings shape: {doc_embeddings.shape}")
        print(f"Random query embeddings shape: {query_embeddings.shape}")
    
    print("\n=== Demo Complete ===")
