"""Modul temu balik semantik (Semantic Retriever) dengan mekanisme ambang batas.

Menyediakan:
- Penilaian kesamaan kosinus (Cosine Similarity).
- Ambang batas penerimaan (similarity threshold) untuk menolak kueri out-of-domain (OOD).
- Penyaringan metadata (metadata filtering) berdasarkan tipe dokumen dan nama berkas sumber.
- Pemrosesan kueri tunggal maupun kelompok (batch retrieval).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from src.knowledge_base.vector_store import VectorStore, SearchResultItem


class QueryStatus(str, Enum):
    """Status hasil evaluasi kueri semantik."""

    ACCEPTED = "accepted"
    REJECTED_OOD = "rejected_out_of_domain"
    EMPTY_QUERY = "empty_query"
    NO_RESULTS = "no_results"


@dataclass(frozen=True)
class RetrievalResult:
    """Representasi hasil temu balik semantik untuk sebuah kueri."""

    query: str
    status: QueryStatus
    items: List[SearchResultItem]
    top_score: float
    threshold: float
    rejection_reason: Optional[str] = None

    @property
    def is_accepted(self) -> bool:
        """Mengembalikan True jika kueri diterima dalam domain kebencanaan."""
        return self.status == QueryStatus.ACCEPTED


class SemanticRetriever:
    """Mesin temu balik semantik cerdas berbasis ChromaDB dengan deteksi OOD."""

    DEFAULT_THRESHOLD = 0.50

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """Inisialisasi SemanticRetriever.

        Args:
            vector_store: Instansi VectorStore yang digunakan untuk pencarian.
                          Jika None, akan menginisialisasi VectorStore default.
            threshold: Ambang batas minimum Cosine Similarity untuk menerima kueri.
        """
        self.vector_store = vector_store or VectorStore()
        self.threshold = float(threshold)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
        source_file: Optional[str] = None,
        min_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """Melakukan temu balik semantik untuk sebuah kueri dengan ambang batas.

        Args:
            query: Teks pertanyaan atau kueri pencarian.
            top_k: Jumlah maksimum potongan dokumen relevan yang dikembalikan.
            doc_type: Filter opsional berdasarkan kategori dokumen.
            source_file: Filter opsional berdasarkan nama berkas PDF sumber.
            min_threshold: Ambang batas spesifik pemanggilan (meng-override default).

        Returns:
            RetrievalResult berisi status kueri, daftar potongan dokumen, dan skor.
        """
        effective_threshold = float(
            min_threshold if min_threshold is not None else self.threshold
        )
        cleaned_query = query.strip() if query else ""

        if not cleaned_query:
            return RetrievalResult(
                query=query,
                status=QueryStatus.EMPTY_QUERY,
                items=[],
                top_score=0.0,
                threshold=effective_threshold,
                rejection_reason="Kueri tidak boleh kosong atau hanya berupa spasi.",
            )

        # Susun klausa filter metadata untuk ChromaDB
        where_filter: Optional[Dict[str, Any]] = None
        filter_conditions: List[Dict[str, Any]] = []

        if doc_type:
            filter_conditions.append({"doc_type": doc_type})
        if source_file:
            filter_conditions.append({"source_file": source_file})

        if len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
        elif len(filter_conditions) > 1:
            where_filter = {"$and": filter_conditions}

        # Eksekusi pencarian vektor di ChromaDB
        raw_results = self.vector_store.query(
            query_text=cleaned_query,
            n_results=top_k,
            where=where_filter,
        )

        if not raw_results:
            return RetrievalResult(
                query=query,
                status=QueryStatus.NO_RESULTS,
                items=[],
                top_score=0.0,
                threshold=effective_threshold,
                rejection_reason="Tidak ditemukan dokumen yang cocok dengan kriteria filter.",
            )

        top_score = raw_results[0].similarity_score

        # Evaluasi ambang batas kesamaan (Similarity Threshold)
        if top_score < effective_threshold:
            return RetrievalResult(
                query=query,
                status=QueryStatus.REJECTED_OOD,
                items=[],
                top_score=top_score,
                threshold=effective_threshold,
                rejection_reason=(
                    f"Kueri berada di luar domain kebencanaan (skor kemiripan tertinggi {top_score:.4f} "
                    f"berada di bawah ambang batas {effective_threshold:.4f})."
                ),
            )

        # Saring hasil agar hanya mengembalikan item yang memenuhi ambang batas
        filtered_items = [
            item for item in raw_results if item.similarity_score >= effective_threshold
        ]

        return RetrievalResult(
            query=query,
            status=QueryStatus.ACCEPTED,
            items=filtered_items,
            top_score=top_score,
            threshold=effective_threshold,
            rejection_reason=None,
        )

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        min_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Melakukan temu balik untuk sekumpulan kueri secara berurutan.

        Args:
            queries: Daftar teks kueri.
            top_k: Jumlah maksimum potongan dokumen per kueri.
            min_threshold: Ambang batas spesifik pemanggilan.

        Returns:
            Daftar RetrievalResult untuk setiap kueri.
        """
        return [
            self.retrieve(q, top_k=top_k, min_threshold=min_threshold)
            for q in queries
        ]

