"""Unit test untuk modul benchmark temu balik semantik (src/benchmark_retrieval.py)."""

from pathlib import Path
import pytest
from src.benchmark_retrieval import TEST_QUERY_SUITE, _generate_similarity_plot


def test_test_query_suite_structure():
    """Memverifikasi bahwa matriks kueri uji bisnis terstruktur lengkap dan valid."""
    assert len(TEST_QUERY_SUITE) >= 10
    
    categories = {q["category"] for q in TEST_QUERY_SUITE}
    assert "Spesifik" in categories
    assert "Parafrase" in categories
    assert "Ambigu" in categories
    assert "Out-of-Domain" in categories

    for q in TEST_QUERY_SUITE:
        assert "id" in q
        assert "category" in q
        assert "query" in q and len(q["query"].strip()) > 0
        assert q["expected_status"] in ("accepted", "rejected_out_of_domain")
        assert "description" in q


def test_generate_similarity_plot(tmp_path):
    """Memverifikasi pembuatan plot diagram batang evaluasi similarity."""
    dummy_results = [
        {"id": "Q1", "category": "Spesifik", "top_similarity_score": 0.82},
        {"id": "Q2", "category": "Parafrase", "top_similarity_score": 0.71},
        {"id": "Q3", "category": "Ambigu", "top_similarity_score": 0.65},
        {"id": "Q4", "category": "Out-of-Domain", "top_similarity_score": 0.32},
    ]
    output_png = tmp_path / "test_similarity.png"
    
    _generate_similarity_plot(dummy_results, threshold=0.50, output_image_path=output_png)
    
    assert output_png.exists()
    assert output_png.stat().st_size > 1000

