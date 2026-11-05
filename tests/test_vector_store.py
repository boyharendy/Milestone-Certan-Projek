"""Unit test untuk penyimpanan vektor persisten ChromaDB (src/knowledge_base/vector_store.py).

Prinsip TDD (Test-Driven Development):
Memvalidasi inisialisasi persisten, indexing batch, query kesamaan kosinus,
filtering metadata, dan persistensi data antar-sesi.
"""

from pathlib import Path
import pytest
from src.knowledge_base.chunker import DocumentChunk
from src.knowledge_base.embedding_service import BaseEmbeddingService, DummyEmbeddingService
from src.knowledge_base.vector_store import VectorStore, SearchResultItem


@pytest.fixture
def dummy_embedding():
    """Embedding tiruan berdimensi 4 untuk pengujian unit yang cepat dan deterministik."""
    return DummyEmbeddingService(dimension=4)


@pytest.fixture
def sample_chunks():
    """Sampel DocumentChunk untuk pengujian indexing."""
    return [
        DocumentChunk(
            chunk_id="sop_tsunami_p1_c0",
            text="Prosedur operasional standar aktivasi sirine peringatan dini tsunami di pesisir Banda Aceh.",
            source_file="sop_tsunami.pdf",
            doc_title="SOP Kesiapsiagaan Tsunami",
            doc_type="SOP_Operasional",
            page_number=1,
            chunk_index=0,
            char_count=95,
            word_count=12,
        ),
        DocumentChunk(
            chunk_id="sop_tsunami_p2_c1",
            text="Pusdalops BPBD mengoordinasikan rute evakuasi warga menuju Tempat Evakuasi Sementara.",
            source_file="sop_tsunami.pdf",
            doc_title="SOP Kesiapsiagaan Tsunami",
            doc_type="SOP_Operasional",
            page_number=2,
            chunk_index=1,
            char_count=85,
            word_count=10,
        ),
        DocumentChunk(
            chunk_id="qanun_pb_p5_c2",
            text="Qanun Aceh menetapkan regulasi kelembagaan dan pendanaan darurat penanggulangan bencana.",
            source_file="qanun_pb.pdf",
            doc_title="Qanun Penanggulangan Bencana",
            doc_type="Regulasi_Hukum",
            page_number=5,
            chunk_index=2,
            char_count=87,
            word_count=10,
        ),
    ]


class TestVectorStore:
    """Pengujian fungsionalitas VectorStore ChromaDB persisten."""

    def test_init_and_count_empty(self, tmp_path, dummy_embedding):
        store = VectorStore(persist_dir=str(tmp_path), embedding_service=dummy_embedding)
        assert store.count() == 0

    def test_add_chunks_and_query(self, tmp_path, dummy_embedding, sample_chunks):
        store = VectorStore(persist_dir=str(tmp_path), embedding_service=dummy_embedding)
        inserted = store.add_chunks(sample_chunks)
        assert inserted == 3
        assert store.count() == 3

        # Uji query temu balik
        results = store.query(query_text="aktivasi sirine tsunami", n_results=2)
        assert len(results) == 2
        assert isinstance(results[0], SearchResultItem)
        assert results[0].chunk_id in [c.chunk_id for c in sample_chunks]
        assert results[0].doc_type in ["SOP_Operasional", "Regulasi_Hukum"]
        assert 0.0 <= results[0].similarity_score <= 1.0

    def test_metadata_filtering(self, tmp_path, dummy_embedding, sample_chunks):
        store = VectorStore(persist_dir=str(tmp_path), embedding_service=dummy_embedding)
        store.add_chunks(sample_chunks)

        # Query dengan filter hanya doc_type = Regulasi_Hukum
        results = store.query(
            query_text="bencana",
            n_results=5,
            where={"doc_type": "Regulasi_Hukum"},
        )
        assert len(results) == 1
        assert results[0].chunk_id == "qanun_pb_p5_c2"
        assert results[0].doc_type == "Regulasi_Hukum"

    def test_persistence_between_sessions(self, tmp_path, dummy_embedding, sample_chunks):
        persist_dir = str(tmp_path / "chroma_db_test")

        # Sesi 1: Tulis data ke disk
        store_session1 = VectorStore(persist_dir=persist_dir, embedding_service=dummy_embedding)
        store_session1.add_chunks(sample_chunks)
        assert store_session1.count() == 3
        del store_session1

        # Sesi 2: Buka kembali direktori yang sama
        store_session2 = VectorStore(persist_dir=persist_dir, embedding_service=dummy_embedding)
        assert store_session2.count() == 3

        results = store_session2.query("Pusdalops evakuasi", n_results=1)
        assert len(results) == 1
        assert results[0].chunk_id in [c.chunk_id for c in sample_chunks]

    def test_reset_store(self, tmp_path, dummy_embedding, sample_chunks):
        store = VectorStore(persist_dir=str(tmp_path), embedding_service=dummy_embedding)
        store.add_chunks(sample_chunks)
        assert store.count() == 3

        store.reset()
        assert store.count() == 0

