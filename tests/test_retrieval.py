"""Unit test untuk modul semantic search retriever (src/knowledge_base/retriever.py).

Prinsip TDD (Test-Driven Development):
Menguji logika temu balik semantik, mekanisme penolakan kueri out-of-domain (OOD)
berbasis similarity threshold, penyaringan metadata (metadata filtering), dan batch retrieval.
"""

from pathlib import Path
import pytest
from src.knowledge_base.chunker import DocumentChunk
from src.knowledge_base.embedding_service import DummyEmbeddingService
from src.knowledge_base.vector_store import VectorStore, SearchResultItem
from src.knowledge_base.retriever import (
    SemanticRetriever,
    RetrievalResult,
    QueryStatus,
    SearchMode,
    compute_lexical_score,
)


@pytest.fixture
def dummy_embedding():
    """Embedding tiruan berdimensi 4 untuk pengujian unit yang cepat dan deterministik."""
    return DummyEmbeddingService(dimension=4)


@pytest.fixture
def sample_chunks():
    """Daftar potongan teks sampel untuk inisialisasi basis data pengujian."""
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
            chunk_id="tes_tsunami_p2_c0",
            text="Standar teknis konstruksi Tempat Evakuasi Sementara TES tsunami memiliki daya tampung 2000 jiwa.",
            source_file="pedoman_tes.pdf",
            doc_title="Pedoman Teknis TES",
            doc_type="Pedoman_Teknis",
            page_number=2,
            chunk_index=0,
            char_count=98,
            word_count=13,
        ),
        DocumentChunk(
            chunk_id="inpres_nad_p1_c0",
            text="Instruksi Presiden Nomor 1 Tahun 2005 tentang kegiatan tanggap darurat pasca bencana tsunami NAD.",
            source_file="In_Pres No.1 Tahun 2005.pdf",
            doc_title="In Pres No.1 Tahun 2005",
            doc_type="Regulasi_Hukum",
            page_number=1,
            chunk_index=0,
            char_count=100,
            word_count=14,
        ),
    ]


@pytest.fixture
def populated_vector_store(tmp_path, dummy_embedding, sample_chunks):
    """VectorStore dengan sampel chunk dokumen kebencanaan."""
    store = VectorStore(persist_dir=str(tmp_path), embedding_service=dummy_embedding)
    store.add_chunks(sample_chunks)
    return store


class TestSemanticRetriever:
    """Suite pengujian unit untuk SemanticRetriever."""

    def test_empty_query(self, populated_vector_store):
        """Kueri kosong atau whitespace harus ditangani dengan status EMPTY_QUERY."""
        retriever = SemanticRetriever(vector_store=populated_vector_store, threshold=0.50)
        
        result_empty = retriever.retrieve("")
        assert result_empty.status == QueryStatus.EMPTY_QUERY
        assert not result_empty.is_accepted
        assert len(result_empty.items) == 0
        assert result_empty.top_score == 0.0

        result_spaces = retriever.retrieve("    ")
        assert result_spaces.status == QueryStatus.EMPTY_QUERY
        assert not result_spaces.is_accepted

    def test_in_domain_query_accepted(self, populated_vector_store, sample_chunks):
        """Kueri dengan kemiripan tinggi (>= threshold) harus diterima (ACCEPTED)."""
        retriever = SemanticRetriever(vector_store=populated_vector_store, threshold=0.50)
        
        # Menggunakan teks yang cocok dengan chunk
        query = sample_chunks[0].text
        result = retriever.retrieve(query, top_k=2)
        assert result.status == QueryStatus.ACCEPTED
        assert result.is_accepted is True
        assert len(result.items) > 0
        assert result.top_score >= 0.50
        assert result.rejection_reason is None

    def test_out_of_domain_query_rejected(self, populated_vector_store):
        """Kueri dengan similarity < threshold harus ditolak (REJECTED_OOD)."""
        # Set threshold tinggi untuk menguji penolakan OOD
        retriever = SemanticRetriever(vector_store=populated_vector_store, threshold=0.99)
        
        result = retriever.retrieve("resep membuat mie aceh kepiting kuah kari non literal")
        assert result.status == QueryStatus.REJECTED_OOD
        assert result.is_accepted is False
        assert result.rejection_reason is not None
        assert "di bawah ambang batas" in result.rejection_reason.lower()

    def test_custom_threshold_override(self, populated_vector_store, sample_chunks):
        """Parameter min_threshold pada pemanggilan fungsi harus meng-override threshold default."""
        retriever = SemanticRetriever(vector_store=populated_vector_store, threshold=0.50)
        query = sample_chunks[0].text
        
        # Override dengan threshold sangat ketat (1.1) -> ditolak
        result_strict = retriever.retrieve(query, min_threshold=1.1)
        assert result_strict.status == QueryStatus.REJECTED_OOD
        assert result_strict.threshold == 1.1

        # Override dengan threshold permisif (0.1) -> diterima
        result_permissive = retriever.retrieve(query, min_threshold=0.1)
        assert result_permissive.status == QueryStatus.ACCEPTED
        assert result_permissive.threshold == 0.1

    def test_metadata_filtering_doc_type(self, populated_vector_store, sample_chunks):
        """Retriever harus menyaring dokumen berdasarkan doc_type."""
        retriever = SemanticRetriever(vector_store=populated_vector_store, threshold=0.0)
        query = sample_chunks[2].text
        
        result = retriever.retrieve(query, doc_type="Regulasi_Hukum", top_k=5)
        assert result.status == QueryStatus.ACCEPTED
        assert len(result.items) > 0
        for item in result.items:
            assert item.doc_type == "Regulasi_Hukum"

    def test_metadata_filtering_source_file(self, populated_vector_store, sample_chunks):
        """Retriever harus menyaring dokumen berdasarkan source_file spesifik."""
        retriever = SemanticRetriever(vector_store=populated_vector_store, threshold=0.0)
        query = sample_chunks[2].text
        target_file = "In_Pres No.1 Tahun 2005.pdf"
        
        result = retriever.retrieve(query, source_file=target_file, top_k=5)
        assert result.status == QueryStatus.ACCEPTED
        assert len(result.items) > 0
        for item in result.items:
            assert item.source_file == target_file

    def test_metadata_filtering_combined(self, populated_vector_store, sample_chunks):
        """Retriever harus mendukung penyaringan kombinasi (doc_type dan source_file)."""
        retriever = SemanticRetriever(vector_store=populated_vector_store, threshold=0.0)
        query = sample_chunks[2].text
        
        result = retriever.retrieve(
            query,
            doc_type="Regulasi_Hukum",
            source_file="In_Pres No.1 Tahun 2005.pdf",
            top_k=5,
        )
        assert result.status == QueryStatus.ACCEPTED
        assert len(result.items) > 0
        for item in result.items:
            assert item.doc_type == "Regulasi_Hukum"
            assert item.source_file == "In_Pres No.1 Tahun 2005.pdf"

    def test_no_results_when_filter_mismatch(self, populated_vector_store, sample_chunks):
        """Retriever harus menghasilkan NO_RESULTS jika filter metadata tidak menghasilkan dokumen cocok."""
        retriever = SemanticRetriever(vector_store=populated_vector_store, threshold=0.0)
        query = sample_chunks[0].text
        
        result = retriever.retrieve(query, doc_type="Kategori_Non_Existent")
        assert result.status == QueryStatus.NO_RESULTS
        assert not result.is_accepted
        assert len(result.items) == 0

    def test_batch_retrieve(self, populated_vector_store, sample_chunks):
        """batch_retrieve harus memproses sekumpulan kueri secara terurut."""
        retriever = SemanticRetriever(vector_store=populated_vector_store, threshold=0.50)
        queries = [
            sample_chunks[0].text,
            "",
            sample_chunks[1].text,
        ]
        
        results = retriever.batch_retrieve(queries, top_k=2)
        assert len(results) == 3
        assert results[0].status == QueryStatus.ACCEPTED
        assert results[0].query == queries[0]
        assert results[1].status == QueryStatus.EMPTY_QUERY
        assert results[2].status == QueryStatus.ACCEPTED
        assert results[2].query == queries[2]

    def test_compute_lexical_score(self):
        """Memverifikasi fungsi penghitungan skor leksikal keyword."""
        query = "aktivasi sirine tsunami"
        doc_match = "Prosedur aktivasi sirine peringatan dini tsunami di pesisir Banda Aceh."
        doc_irrelevant = "Resep membuat mie aceh tradisional bumbu kepiting kuah kari."

        score_match = compute_lexical_score(query, doc_match)
        score_irrelevant = compute_lexical_score(query, doc_irrelevant)

        assert score_match > 0.50
        assert score_irrelevant == 0.0
        assert compute_lexical_score("", doc_match) == 0.0

    def test_search_modes(self, populated_vector_store, sample_chunks):
        """Memverifikasi ketiga mode pencarian: semantic, lexical, dan hybrid."""
        retriever = SemanticRetriever(vector_store=populated_vector_store, threshold=0.10)
        query = "aktivasi sirine tsunami"

        # Mode Semantic
        res_sem = retriever.retrieve(query, mode=SearchMode.SEMANTIC)
        assert res_sem.mode == SearchMode.SEMANTIC
        assert res_sem.status == QueryStatus.ACCEPTED

        # Mode Lexical
        res_lex = retriever.retrieve(query, mode=SearchMode.LEXICAL)
        assert res_lex.mode == SearchMode.LEXICAL
        assert res_lex.status == QueryStatus.ACCEPTED
        assert res_lex.items[0].source_file == "sop_tsunami.pdf"

        # Mode Hybrid (Default)
        res_hyb = retriever.retrieve(query, mode=SearchMode.HYBRID, alpha=0.70)
        assert res_hyb.mode == SearchMode.HYBRID
        assert res_hyb.status == QueryStatus.ACCEPTED
        assert len(res_hyb.items) > 0
