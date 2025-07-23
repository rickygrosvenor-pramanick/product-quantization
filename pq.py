"""
Product Quantization (PQ) Module

This module implements Product Quantization for approximate nearest neighbor search
and vector compression. It provides functionality to:
- Split high-dimensional vectors into sub-spaces
- Learn codebooks using k-means clustering
- Encode/decode vectors using learned quantizers
- Build ANN index for fast similarity search
- Evaluate performance metrics

Author: Ricky Pramanick
Date: July 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from sklearn.cluster import KMeans
import hnswlib


class ProductQuantizer:
    """
    Product Quantization implementation for vector compression and approximate search.
    
    Product Quantization splits D-dimensional vectors into M sub-vectors of dimension D/M,
    then quantizes each sub-space independently using k-means clustering with K centroids.
    
    Attributes:
        M (int): Number of sub-spaces
        K (int): Number of centroids per codebook
        D (int): Original vector dimension
        sub_dim (int): Dimension of each sub-vector (D/M)
        codebooks (np.ndarray): Learned centroids for each sub-space
        is_trained (bool): Whether the quantizer has been trained
    """
    
    def __init__(self, M: int, K: int, verbose: bool = False):
        """
        Initialize Product Quantizer.
        
        Args:
            M (int): Number of sub-spaces to split vectors into
            K (int): Number of centroids per codebook (typically 256 for 8-bit codes)
            verbose (bool): Whether to print training progress
        """
        # Validate parameters
        if M <= 0:
            raise ValueError("Number of sub-spaces M must be positive")
        if K <= 0:
            raise ValueError("Number of centroids K must be positive")
        if K > 65536:
            raise ValueError("K too large, max supported is 65536 for uint16 codes")
        
        # Core parameters
        self.M = M  # Number of sub-spaces
        self.K = K  # Number of centroids per codebook
        self.verbose = verbose
        
        # Will be set during training
        self.D = None  # Original vector dimension
        self.sub_dim = None  # Dimension of each sub-vector (D/M)
        self.codebooks = None  # Learned centroids for each sub-space, shape (M, K, sub_dim)
        self.is_trained = False
        
        # Determine appropriate dtype for codes based on K
        if K <= 256:
            self.code_dtype = np.uint8
        else:
            self.code_dtype = np.uint16
    
    def fit(self, embeddings: np.ndarray, n_iter: int = 20, random_state: int = 42) -> 'ProductQuantizer':
        """
        Train the product quantizer on input embeddings.
        
        Splits the D-dimensional vectors into M sub-vectors and runs k-means
        clustering on each sub-space to learn K centroids per codebook.
        
        Args:
            embeddings (np.ndarray): Input vectors of shape (N, D)
            n_iter (int): Maximum iterations for k-means
            random_state (int): Random seed for reproducibility
            
        Returns:
            ProductQuantizer: Self for method chaining
            
        Raises:
            ValueError: If embeddings shape is invalid or D not divisible by M
        """
        # Validate input
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D")
        
        N, D = embeddings.shape
        
        if N < self.K:
            raise ValueError(f"Need at least {self.K} vectors to learn {self.K} centroids, got {N}")
        
        if D % self.M != 0:
            raise ValueError(f"Dimension {D} must be divisible by M={self.M}")
        
        # Set dimensions
        self.D = D
        self.sub_dim = D // self.M
        
        if self.verbose:
            print(f"Training PQ with {N} vectors of dimension {D}")
            print(f"Sub-spaces: {self.M}, Sub-dimension: {self.sub_dim}, Centroids per codebook: {self.K}")
        
        # Initialize codebooks array: shape (M, K, sub_dim)
        self.codebooks = np.zeros((self.M, self.K, self.sub_dim), dtype=np.float32)
        
        # Train k-means for each sub-space
        for m in range(self.M):
            if self.verbose:
                print(f"Training codebook {m+1}/{self.M}...")
            
            # Extract sub-vectors for this sub-space
            start_idx = m * self.sub_dim
            end_idx = (m + 1) * self.sub_dim
            sub_vectors = embeddings[:, start_idx:end_idx]
            
            # Run k-means clustering
            kmeans = KMeans(
                n_clusters=self.K,
                max_iter=n_iter,
                random_state=random_state + m,  # Different seed per sub-space
                n_init=1,  # Single run since we control random_state
                algorithm='lloyd'  # Use standard Lloyd's algorithm
            )
            
            # Fit k-means and store centroids
            kmeans.fit(sub_vectors)
            self.codebooks[m] = kmeans.cluster_centers_.astype(np.float32)
            
            if self.verbose:
                inertia = kmeans.inertia_
                print(f"  Codebook {m+1} inertia: {inertia:.3f}")
        
        self.is_trained = True
        
        if self.verbose:
            # Calculate some training statistics
            total_params = self.M * self.K * self.sub_dim
            original_size = N * D * 4  # float32
            codebook_size = total_params * 4  # float32
            print(f"Training complete!")
            print(f"Codebook parameters: {total_params:,}")
            print(f"Codebook size: {codebook_size / 1024:.1f} KB")
            
        return self
    
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode vectors into PQ codes.
        
        Maps each sub-vector to its nearest centroid index, producing
        compact codes of shape (N, M) with values in [0, K-1].
        
        Args:
            embeddings (np.ndarray): Input vectors of shape (N, D)
            
        Returns:
            np.ndarray: PQ codes of shape (N, M) with dtype uint8/uint16
            
        Raises:
            RuntimeError: If quantizer not trained yet
            ValueError: If input dimension doesn't match training data
        """
        # Check if quantizer is trained
        if not self.is_trained:
            raise RuntimeError("Quantizer must be trained before encoding. Call fit() first.")
        
        # Validate input
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D")
        
        N, D = embeddings.shape
        
        if D != self.D:
            raise ValueError(f"Input dimension {D} doesn't match training dimension {self.D}")
        
        # Initialize codes array
        codes = np.zeros((N, self.M), dtype=self.code_dtype)
        
        # Encode each sub-space independently
        for m in range(self.M):
            # Extract sub-vectors for this sub-space
            start_idx = m * self.sub_dim
            end_idx = (m + 1) * self.sub_dim
            sub_vectors = embeddings[:, start_idx:end_idx]  # Shape: (N, sub_dim)
            
            # Get centroids for this sub-space
            centroids = self.codebooks[m]  # Shape: (K, sub_dim)
            
            # Compute distances from each sub-vector to all centroids
            # Using broadcasting: (N, 1, sub_dim) - (1, K, sub_dim) = (N, K, sub_dim)
            distances = np.linalg.norm(
                sub_vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :], 
                axis=2
            )  # Shape: (N, K)
            
            # Find nearest centroid for each sub-vector
            nearest_centroids = np.argmin(distances, axis=1)  # Shape: (N,)
            
            # Store the centroid indices as codes
            codes[:, m] = nearest_centroids.astype(self.code_dtype)
        
        if self.verbose:
            # Calculate compression statistics
            original_size = embeddings.nbytes
            codes_size = codes.nbytes
            compression_ratio = original_size / codes_size
            print(f"Encoded {N} vectors: {original_size} bytes → {codes_size} bytes")
            print(f"Compression ratio: {compression_ratio:.1f}x")
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode PQ codes back to approximate embeddings.
        
        Reconstructs vectors by looking up centroids for each code
        and concatenating sub-vectors.
        
        Args:
            codes (np.ndarray): PQ codes of shape (N, M)
            
        Returns:
            np.ndarray: Reconstructed vectors of shape (N, D)
            
        Raises:
            RuntimeError: If quantizer not trained yet
            ValueError: If codes contain invalid indices
        """
        # Check if quantizer is trained
        if not self.is_trained:
            raise RuntimeError("Quantizer must be trained before decoding. Call fit() first.")
        
        # Validate input
        if codes.ndim != 2:
            raise ValueError(f"Codes must be 2D array, got {codes.ndim}D")
        
        N, M = codes.shape
        
        if M != self.M:
            raise ValueError(f"Number of codes {M} doesn't match number of sub-spaces {self.M}")
        
        # Check for invalid indices
        if np.any(codes >= self.K) or np.any(codes < 0):
            raise ValueError(f"Codes contain invalid indices. Must be in range [0, {self.K-1}]")
        
        # Initialize reconstructed embeddings
        reconstructed = np.zeros((N, self.D), dtype=np.float32)
        
        # Decode each sub-space independently
        for m in range(self.M):
            # Get codes for this sub-space
            sub_codes = codes[:, m]  # Shape: (N,)
            
            # Look up centroids for these codes
            sub_centroids = self.codebooks[m][sub_codes]  # Shape: (N, sub_dim)
            
            # Place reconstructed sub-vectors in the correct positions
            start_idx = m * self.sub_dim
            end_idx = (m + 1) * self.sub_dim
            reconstructed[:, start_idx:end_idx] = sub_centroids
        
        return reconstructed
    
    def compute_distance_table(self, query: np.ndarray) -> np.ndarray:
        """
        Compute asymmetric distance table for a query vector.
        
        Pre-computes distances from query sub-vectors to all centroids
        in each codebook, enabling fast distance computation to PQ codes.
        
        Args:
            query (np.ndarray): Query vector of shape (D,)
            
        Returns:
            np.ndarray: Distance table of shape (M, K)
            
        Raises:
            RuntimeError: If quantizer not trained yet
        """
        # Check if quantizer is trained
        if not self.is_trained:
            raise RuntimeError("Quantizer must be trained before computing distance table. Call fit() first.")
        
        # Validate input
        if query.ndim != 1:
            raise ValueError(f"Query must be 1D array, got {query.ndim}D")
        
        if len(query) != self.D:
            raise ValueError(f"Query dimension {len(query)} doesn't match training dimension {self.D}")
        
        # Initialize distance table
        distance_table = np.zeros((self.M, self.K), dtype=np.float32)
        
        # Compute distances for each sub-space
        for m in range(self.M):
            # Extract query sub-vector for this sub-space
            start_idx = m * self.sub_dim
            end_idx = (m + 1) * self.sub_dim
            query_sub = query[start_idx:end_idx]  # Shape: (sub_dim,)
            
            # Get centroids for this sub-space
            centroids = self.codebooks[m]  # Shape: (K, sub_dim)
            
            # Compute distances from query sub-vector to all centroids
            distances = np.linalg.norm(centroids - query_sub[np.newaxis, :], axis=1)  # Shape: (K,)
            
            # Store distances
            distance_table[m] = distances
        
        return distance_table
    
    def asymmetric_distance(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """
        Compute asymmetric distances from query to PQ codes.
        
        Uses pre-computed distance table for efficient distance computation
        between original query vector and quantized database vectors.
        
        Args:
            query (np.ndarray): Query vector of shape (D,)
            codes (np.ndarray): PQ codes of shape (N, M)
            
        Returns:
            np.ndarray: Distances of shape (N,)
        """
        # Compute distance table once
        distance_table = self.compute_distance_table(query)
        
        # Validate codes
        if codes.ndim != 2:
            raise ValueError(f"Codes must be 2D array, got {codes.ndim}D")
        
        N, M = codes.shape
        if M != self.M:
            raise ValueError(f"Number of codes {M} doesn't match number of sub-spaces {self.M}")
        
        # Sum squared distances across sub-spaces
        total_distances = np.zeros(N, dtype=np.float32)
        
        for m in range(self.M):
            # Get codes for this sub-space
            sub_codes = codes[:, m]  # Shape: (N,)
            
            # Look up distances from distance table
            sub_distances = distance_table[m, sub_codes]  # Shape: (N,)
            
            # Add squared distances (for L2 norm)
            total_distances += sub_distances ** 2
        
        # Return L2 distances (square root of sum of squared distances)
        return np.sqrt(total_distances)
    
    def get_memory_usage(self, n_vectors: int) -> Dict[str, float]:
        """
        Calculate memory usage statistics.
        
        Computes memory requirements for storing vectors in different formats:
        original float32, PQ codes, and codebooks.
        
        Args:
            n_vectors (int): Number of vectors to analyze
            
        Returns:
            Dict[str, float]: Memory usage in bytes for different representations
        """
        if not self.is_trained:
            # Estimate based on parameters
            D = 384  # Default embedding dimension
            codebook_size = self.M * self.K * (D // self.M) * 4  # float32
        else:
            D = self.D
            codebook_size = self.codebooks.nbytes
        
        # Calculate sizes
        original_size = n_vectors * D * 4  # float32
        codes_size = n_vectors * self.M * (1 if self.K <= 256 else 2)  # uint8 or uint16
        total_pq_size = codes_size + codebook_size
        
        return {
            'original_vectors': float(original_size),
            'pq_codes': float(codes_size),
            'codebooks': float(codebook_size),
            'total_pq': float(total_pq_size),
            'compression_ratio': float(original_size / total_pq_size) if total_pq_size > 0 else 0.0
        }
    
    def save(self, filepath: str) -> None:
        """
        Save trained quantizer to disk.
        
        Args:
            filepath (str): Path to save the quantizer
            
        Raises:
            RuntimeError: If quantizer not trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained quantizer. Call fit() first.")
        
        save_data = {
            'M': self.M,
            'K': self.K,
            'D': self.D,
            'sub_dim': self.sub_dim,
            'codebooks': self.codebooks,
            'code_dtype': self.code_dtype,
            'is_trained': self.is_trained
        }
        
        np.savez_compressed(filepath, **save_data)
        
        if self.verbose:
            print(f"Saved quantizer to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ProductQuantizer':
        """
        Load trained quantizer from disk.
        
        Args:
            filepath (str): Path to load the quantizer from
            
        Returns:
            ProductQuantizer: Loaded quantizer instance
        """
        data = np.load(filepath, allow_pickle=True)
        
        # Create new instance
        pq = cls(M=int(data['M']), K=int(data['K']))
        
        # Restore state
        pq.D = int(data['D'])
        pq.sub_dim = int(data['sub_dim'])
        pq.codebooks = data['codebooks']
        pq.code_dtype = data['code_dtype'].item()
        pq.is_trained = bool(data['is_trained'])
        
        return pq


class PQIndex:
    """
    Approximate Nearest Neighbor index using Product Quantization.
    
    Combines PQ encoding with HNSW for fast similarity search over
    compressed vectors using asymmetric distance computation.
    """
    
    def __init__(self, pq: ProductQuantizer, max_elements: int = 1000000, 
                 ef_construction: int = 200, M_hnsw: int = 16):
        """
        Initialize PQ-based ANN index.
        
        Args:
            pq (ProductQuantizer): Trained product quantizer
            max_elements (int): Maximum number of vectors to index
            ef_construction (int): HNSW construction parameter
            M_hnsw (int): HNSW connectivity parameter
            
        Raises:
            RuntimeError: If product quantizer not trained
        """
        pass
    
    def add_vectors(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """
        Add vectors to the index.
        
        Encodes vectors using PQ and adds them to the HNSW index
        for fast approximate search.
        
        Args:
            embeddings (np.ndarray): Vectors to add, shape (N, D)
            ids (Optional[np.ndarray]): Optional vector IDs, shape (N,)
        """
        pass
    
    def search(self, query: np.ndarray, k: int = 10, ef: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for approximate nearest neighbors.
        
        Uses asymmetric distance computation between original query
        and PQ-encoded database vectors.
        
        Args:
            query (np.ndarray): Query vector of shape (D,)
            k (int): Number of nearest neighbors to return
            ef (int): HNSW search parameter
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, indices) of nearest neighbors
        """
        pass
    
    def batch_search(self, queries: np.ndarray, k: int = 10, ef: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries.
        
        Args:
            queries (np.ndarray): Query vectors of shape (N_queries, D)
            k (int): Number of nearest neighbors per query
            ef (int): HNSW search parameter
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, indices) for all queries
        """
        pass


def generate_random_embeddings(n_vectors: int, dimension: int, 
                             random_state: int = 42) -> np.ndarray:
    """
    Generate random embeddings for testing.
    
    Creates normalized random vectors that simulate real embeddings
    for benchmarking and testing purposes.
    
    Args:
        n_vectors (int): Number of vectors to generate
        dimension (int): Vector dimension
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


def evaluate_recall(true_neighbors: np.ndarray, predicted_neighbors: np.ndarray, 
                   k_values: List[int]) -> Dict[int, float]:
    """
    Evaluate recall@K for approximate search results.
    
    Computes recall at different K values by comparing approximate
    search results with ground truth nearest neighbors.
    
    Args:
        true_neighbors (np.ndarray): Ground truth neighbors, shape (N_queries, K_max)
        predicted_neighbors (np.ndarray): Predicted neighbors, shape (N_queries, K_max)
        k_values (List[int]): K values to evaluate recall for
        
    Returns:
        Dict[int, float]: Recall@K for each K value
    """
    recall_results = {}
    n_queries = true_neighbors.shape[0]
    
    for k in k_values:
        if k > true_neighbors.shape[1] or k > predicted_neighbors.shape[1]:
            recall_results[k] = 0.0
            continue
        
        total_recall = 0.0
        
        for i in range(n_queries):
            true_set = set(true_neighbors[i, :k])
            pred_set = set(predicted_neighbors[i, :k])
            
            # Calculate recall for this query
            intersection = len(true_set & pred_set)
            recall = intersection / len(true_set) if len(true_set) > 0 else 0.0
            total_recall += recall
        
        recall_results[k] = total_recall / n_queries
    
    return recall_results


def benchmark_search_latency(index, queries: np.ndarray, 
                           k: int = 10, n_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark search latency performance.
    
    Measures query latency statistics for the PQ index including
    mean, median, and percentile latencies.
    
    Args:
        index: PQ index to benchmark (SimpleRetrievalSystem or PQIndex)
        queries (np.ndarray): Query vectors for benchmarking
        k (int): Number of neighbors to retrieve
        n_runs (int): Number of benchmark runs
        
    Returns:
        Dict[str, float]: Latency statistics in milliseconds
    """
    import time
    
    latencies = []
    n_queries = min(len(queries), n_runs)
    
    # Warm up
    if hasattr(index, 'search'):
        for i in range(min(5, n_queries)):
            index.search(queries[i], k=k)
    
    # Benchmark
    for i in range(n_queries):
        query = queries[i % len(queries)]
        
        start_time = time.perf_counter()
        if hasattr(index, 'search'):
            index.search(query, k=k)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'std_ms': float(np.std(latencies))
    }


def compare_compression_ratios(embeddings: np.ndarray, M_values: List[int], 
                             K_values: List[int]) -> Dict[str, Any]:
    """
    Compare compression ratios for different PQ parameters.
    
    Analyzes memory usage and reconstruction error for various
    combinations of M (sub-spaces) and K (centroids) values.
    
    Args:
        embeddings (np.ndarray): Sample embeddings for analysis
        M_values (List[int]): Sub-space values to test
        K_values (List[int]): Centroid values to test
        
    Returns:
        Dict[str, Any]: Compression analysis results
    """
    results = {
        'configurations': [],
        'compression_ratios': [],
        'reconstruction_errors': [],
        'memory_usage': []
    }
    
    n_samples = min(1000, len(embeddings))  # Use subset for faster analysis
    sample_embeddings = embeddings[:n_samples]
    
    for M in M_values:
        for K in K_values:
            # Check if dimension is divisible by M
            if embeddings.shape[1] % M != 0:
                continue
                
            try:
                # Train PQ
                pq = ProductQuantizer(M=M, K=K, verbose=False)
                pq.fit(sample_embeddings)
                
                # Encode and decode
                codes = pq.encode(sample_embeddings)
                reconstructed = pq.decode(codes)
                
                # Calculate reconstruction error (MSE)
                mse = np.mean((sample_embeddings - reconstructed) ** 2)
                
                # Get memory usage
                memory_stats = pq.get_memory_usage(n_samples)
                
                results['configurations'].append(f"M={M}, K={K}")
                results['compression_ratios'].append(memory_stats['compression_ratio'])
                results['reconstruction_errors'].append(float(mse))
                results['memory_usage'].append(memory_stats)
                
            except Exception as e:
                print(f"Failed for M={M}, K={K}: {e}")
                continue
    
    return results


def brute_force_search(embeddings: np.ndarray, queries: np.ndarray, 
                      k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brute force exact nearest neighbor search.
    
    Computes exact nearest neighbors for ground truth comparison
    with approximate search methods.
    
    Args:
        embeddings (np.ndarray): Database vectors, shape (N, D)
        queries (np.ndarray): Query vectors, shape (N_queries, D)
        k (int): Number of neighbors to return
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (distances, indices) of exact neighbors
    """
    n_queries = queries.shape[0]
    n_embeddings = embeddings.shape[0]
    
    # Ensure k doesn't exceed number of embeddings
    k = min(k, n_embeddings)
    
    all_distances = np.zeros((n_queries, k), dtype=np.float32)
    all_indices = np.zeros((n_queries, k), dtype=np.int32)
    
    for i, query in enumerate(queries):
        # Compute distances to all embeddings
        distances = np.linalg.norm(embeddings - query[np.newaxis, :], axis=1)
        
        # Get top-k nearest neighbors
        top_k_indices = np.argpartition(distances, k)[:k]
        top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]
        
        all_distances[i] = distances[top_k_indices]
        all_indices[i] = top_k_indices
    
    return all_distances, all_indices


# Utility functions for visualization and analysis

def plot_recall_curve(recall_results: Dict[int, float], title: str = "Recall@K Performance") -> None:
    """
    Plot recall@K curve.
    
    Args:
        recall_results (Dict[int, float]): Recall values for different K
        title (str): Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        k_values = sorted(recall_results.keys())
        recall_values = [recall_results[k] for k in k_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, recall_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('K (Number of Retrieved Items)')
        plt.ylabel('Recall@K')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        # Add value labels on points
        for k, recall in zip(k_values, recall_values):
            plt.annotate(f'{recall:.3f}', (k, recall), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        print("Recall results:")
        for k, recall in sorted(recall_results.items()):
            print(f"  Recall@{k}: {recall:.3f}")


def plot_compression_analysis(compression_results: Dict[str, Any]) -> None:
    """
    Visualize compression ratio vs reconstruction error trade-offs.
    
    Args:
        compression_results (Dict[str, Any]): Results from compare_compression_ratios
    """
    try:
        import matplotlib.pyplot as plt
        
        configs = compression_results['configurations']
        ratios = compression_results['compression_ratios']
        errors = compression_results['reconstruction_errors']
        
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        scatter = plt.scatter(ratios, errors, s=100, alpha=0.7, c=range(len(configs)), cmap='viridis')
        
        # Add labels for each point
        for i, config in enumerate(configs):
            plt.annotate(config, (ratios[i], errors[i]), 
                        textcoords="offset points", xytext=(5,5), fontsize=8)
        
        plt.xlabel('Compression Ratio')
        plt.ylabel('Reconstruction Error (MSE)')
        plt.title('Compression Ratio vs Reconstruction Error Trade-off')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Configuration Index')
        
        # Log scale for better visualization
        plt.xscale('log')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        print("Compression analysis results:")
        configs = compression_results['configurations']
        ratios = compression_results['compression_ratios']
        errors = compression_results['reconstruction_errors']
        
        for config, ratio, error in zip(configs, ratios, errors):
            print(f"  {config}: {ratio:.1f}x compression, {error:.6f} MSE")


def plot_latency_distribution(latency_stats: Dict[str, float]) -> None:
    """
    Plot search latency distribution.
    
    Args:
        latency_stats (Dict[str, float]): Latency statistics from benchmark
    """
    try:
        import matplotlib.pyplot as plt
        
        # Extract statistics
        stats = ['mean_ms', 'median_ms', 'p95_ms', 'p99_ms', 'max_ms']
        labels = ['Mean', 'Median', 'P95', 'P99', 'Max']
        values = [latency_stats.get(stat, 0) for stat in stats]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=['skyblue', 'lightgreen', 'orange', 'red', 'darkred'])
        
        plt.xlabel('Latency Metric')
        plt.ylabel('Latency (ms)')
        plt.title('Search Latency Distribution')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        print("Latency statistics:")
        for key, value in latency_stats.items():
            print(f"  {key}: {value:.3f} ms")


if __name__ == "__main__":
    """
    Example usage and basic testing of the PQ module.
    """
    print("=== Product Quantization Module Demo ===")
    
    # Generate sample data
    print("\n1. Generating sample embeddings...")
    n_vectors = 1000
    dimension = 384  # Cohere light model dimension
    
    embeddings = generate_random_embeddings(n_vectors, dimension, random_state=42)
    print(f"Generated {n_vectors} embeddings of dimension {dimension}")
    
    # Train Product Quantizer
    print("\n2. Training Product Quantizer...")
    pq = ProductQuantizer(M=8, K=256, verbose=True)
    pq.fit(embeddings)
    
    # Encode embeddings
    print("\n3. Encoding embeddings...")
    codes = pq.encode(embeddings)
    print(f"Encoded embeddings shape: {codes.shape}")
    print(f"Codes dtype: {codes.dtype}")
    
    # Decode embeddings
    print("\n4. Decoding embeddings...")
    reconstructed = pq.decode(codes)
    print(f"Reconstructed embeddings shape: {reconstructed.shape}")
    
    # Calculate reconstruction error
    mse = np.mean((embeddings - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Memory usage analysis
    print("\n5. Memory usage analysis...")
    memory_stats = pq.get_memory_usage(n_vectors)
    print(f"Original vectors: {memory_stats['original_vectors'] / 1024:.1f} KB")
    print(f"PQ codes: {memory_stats['pq_codes'] / 1024:.1f} KB")
    print(f"Codebooks: {memory_stats['codebooks'] / 1024:.1f} KB")
    print(f"Total PQ: {memory_stats['total_pq'] / 1024:.1f} KB")
    print(f"Compression ratio: {memory_stats['compression_ratio']:.1f}x")
    
    # Test search functionality
    print("\n6. Testing asymmetric distance computation...")
    query = embeddings[0]  # Use first embedding as query
    distances = pq.asymmetric_distance(query, codes)
    print(f"Computed distances to {len(distances)} vectors")
    
    # Find nearest neighbors
    k = 5
    nearest_indices = np.argsort(distances)[:k]
    print(f"Top {k} nearest neighbors: {nearest_indices}")
    print(f"Distances: {distances[nearest_indices]}")
    
    # Test save/load
    print("\n7. Testing save/load functionality...")
    pq.save("test_pq.npz")
    loaded_pq = ProductQuantizer.load("test_pq.npz")
    print(f"Loaded PQ: M={loaded_pq.M}, K={loaded_pq.K}, D={loaded_pq.D}")
    
    # Compare compression ratios
    print("\n8. Comparing different PQ configurations...")
    M_values = [4, 8, 16]
    K_values = [128, 256]
    
    # Use smaller sample for faster comparison
    sample_embeddings = embeddings[:200]
    comparison_results = compare_compression_ratios(sample_embeddings, M_values, K_values)
    
    print("Configuration comparison:")
    configs = comparison_results['configurations']
    ratios = comparison_results['compression_ratios']
    errors = comparison_results['reconstruction_errors']
    
    for config, ratio, error in zip(configs, ratios, errors):
        print(f"  {config}: {ratio:.1f}x compression, {error:.6f} MSE")
    
    print("\n=== Demo Complete ===")
    print("Your Product Quantization module is ready to use!")
    print("\nNext steps:")
    print("1. Run 'python data_processor.py' to process your data.csv")
    print("2. Use the SimpleRetrievalSystem for fast similarity search")
    print("3. Experiment with different M and K values for your use case")
