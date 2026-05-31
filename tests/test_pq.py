import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pq import (
    ProductQuantizer,
    PQIndex,
    generate_random_embeddings,
    brute_force_search,
)


@pytest.fixture
def embeddings():
    return generate_random_embeddings(500, 384)


@pytest.fixture
def trained_pq(embeddings):
    pq = ProductQuantizer(M=8, K=256)
    pq.fit(embeddings)
    return pq


class TestProductQuantizer:
    def test_fit_sets_dimensions(self, trained_pq):
        assert trained_pq.D == 384
        assert trained_pq.sub_dim == 48
        assert trained_pq.is_trained

    def test_encode_shape_and_dtype(self, trained_pq, embeddings):
        codes = trained_pq.encode(embeddings)
        assert codes.shape == (500, 8)
        assert codes.dtype == np.uint8

    def test_decode_shape(self, trained_pq, embeddings):
        codes = trained_pq.encode(embeddings)
        reconstructed = trained_pq.decode(codes)
        assert reconstructed.shape == embeddings.shape

    def test_reconstruction_error_bounded(self, trained_pq, embeddings):
        codes = trained_pq.encode(embeddings)
        reconstructed = trained_pq.decode(codes)
        mse = float(np.mean((embeddings - reconstructed) ** 2))
        assert mse < 0.01

    def test_asymmetric_distance_self_is_smallest(self, trained_pq, embeddings):
        codes = trained_pq.encode(embeddings)
        dists = trained_pq.asymmetric_distance(embeddings[0], codes)
        assert np.argmin(dists) == 0

    def test_distance_table_shape(self, trained_pq, embeddings):
        table = trained_pq.compute_distance_table(embeddings[0])
        assert table.shape == (8, 256)

    def test_invalid_m_raises(self):
        with pytest.raises(ValueError):
            ProductQuantizer(M=0, K=256)

    def test_dimension_mismatch_raises(self, trained_pq):
        bad = np.random.randn(10, 128).astype(np.float32)
        with pytest.raises(ValueError):
            trained_pq.encode(bad)

    def test_encode_before_fit_raises(self):
        pq = ProductQuantizer(M=8, K=256)
        with pytest.raises(RuntimeError):
            pq.encode(np.random.randn(10, 384).astype(np.float32))

    def test_memory_usage(self, trained_pq):
        stats = trained_pq.get_memory_usage(1000)
        assert stats['compression_ratio'] > 1.0
        assert stats['original_vectors'] > stats['total_pq']

    def test_save_load_roundtrip(self, trained_pq, embeddings, tmp_path):
        path = str(tmp_path / "pq.npz")
        trained_pq.save(path)
        loaded = ProductQuantizer.load(path)
        codes_orig = trained_pq.encode(embeddings[:10])
        codes_loaded = loaded.encode(embeddings[:10])
        np.testing.assert_array_equal(codes_orig, codes_loaded)


class TestPQIndex:
    def test_search_returns_correct_shape(self, trained_pq, embeddings):
        index = PQIndex(trained_pq, max_elements=1000)
        index.add_vectors(embeddings)
        dists, ids = index.search(embeddings[0], k=5)
        assert len(dists) == 5
        assert len(ids) == 5

    def test_self_search_top_result(self, trained_pq, embeddings):
        index = PQIndex(trained_pq, max_elements=1000)
        index.add_vectors(embeddings)
        _, ids = index.search(embeddings[0], k=1)
        assert ids[0] == 0

    def test_batch_search_shape(self, trained_pq, embeddings):
        index = PQIndex(trained_pq, max_elements=1000)
        index.add_vectors(embeddings)
        dists, ids = index.batch_search(embeddings[:5], k=3)
        assert dists.shape == (5, 3)
        assert ids.shape == (5, 3)

    def test_untrained_pq_raises(self):
        pq = ProductQuantizer(M=8, K=256)
        with pytest.raises(RuntimeError):
            PQIndex(pq)


class TestBruteForce:
    def test_self_is_nearest(self, embeddings):
        queries = embeddings[:5]
        _, indices = brute_force_search(embeddings, queries, k=1)
        for i in range(5):
            assert indices[i, 0] == i
