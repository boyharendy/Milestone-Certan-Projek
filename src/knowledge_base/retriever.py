"""Modul temu balik semantik dan pencarian hibrida (True Hybrid Search) dengan mekanisme ambang batas.

Menyediakan:
- Penilaian kesamaan kosinus dense vector (Sentence-Transformers Cosine Similarity).
- Penilaian leksikal berbasis frekuensi kata kunci (Sparse / Lexical Keyword Search).
- Penggabungan skor hibrida terbobot (Hybrid Scoring: alpha * Dense + (1 - alpha) * Lexical).
- Mode pencarian fleksibel: "hybrid" (default), "semantic", dan "lexical".
- Ambang batas penerimaan (confidence thresholding) untuk menolak kueri out-of-domain (OOD).
- Penyaringan metadata (metadata filtering) berdasarkan tipe dokumen dan nama berkas sumber.
- Pemrosesan kueri tunggal maupun kelompok (batch retrieval).
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Dict, List, Optional
from src.knowledge_base.vector_store import VectorStore, SearchResultItem


class QueryStatus(str, Enum):
    """Status hasil evaluasi kueri semantik."""

    ACCEPTED = "accepted"
    REJECTED_OOD = "rejected_out_of_domain"
    EMPTY_QUERY = "empty_query"
    NO_RESULTS = "no_results"


class SearchMode(str, Enum):
    """Mode pencarian temu balik."""

    HYBRID = "hybrid"      # Dense semantic + Lexical keyword matching (Default)
    SEMANTIC = "semantic"  # 100% Dense vector similarity
    LEXICAL = "lexical"    # 100% Sparse keyword matching


@dataclass(frozen=True)
class RetrievalResult:
    """Representasi hasil temu balik semantik/hibrida untuk sebuah kueri."""

    query: str
    status: QueryStatus
    items: List[SearchResultItem]
    top_score: float
    threshold: float
    mode: SearchMode = SearchMode.HYBRID
    rejection_reason: Optional[str] = None

    @property
    def is_accepted(self) -> bool:
        """Mengembalikan True jika kueri diterima dalam domain kebencanaan."""
        return self.status == QueryStatus.ACCEPTED


def compute_lexical_score(query: str, document_text: str) -> float:
    """Menghitung skor kecocokan leksikal berbasis token overlap dan term frequency.

    Args:
        query: Teks kueri pencarian.
        document_text: Teks dokumen atau potongan chunk.

    Returns:
        Skor float normalisasi dalam rentang [0.0, 1.0].
    """
    if not query or not document_text:
        return 0.0

    query_tokens = [
        t for t in re.findall(r"\b\w+\b", query.lower()) if len(t) >= 2
    ]
    if not query_tokens:
        return 0.0

    doc_tokens = re.findall(r"\b\w+\b", document_text.lower())
    if not doc_tokens:
        return 0.0

    doc_counter = Counter(doc_tokens)
    total_doc_tokens = len(doc_tokens)

    # 1. Rasio overlap kata kunci kueri yang ditemukan di dokumen
    unique_query_tokens = set(query_tokens)
    matched_tokens = sum(1 for t in unique_query_tokens if t in doc_counter)
    overlap_ratio = matched_tokens / len(unique_query_tokens)

    # 2. Term frequency contribution dengan normalisasi panjang dokumen
    tf_sum = sum(doc_counter[t] for t in unique_query_tokens if t in doc_counter)
    tf_normalized = tf_sum / (total_doc_tokens ** 0.5)

    # Kombinasi bobot overlap (70%) dan frekuensi kemunculan (30%)
    raw_score = (0.70 * overlap_ratio) + (0.30 * min(1.0, tf_normalized / 2.0))
    return round(max(0.0, min(1.0, raw_score)), 4)


class SemanticRetriever:
    """Mesin temu balik semantik dan hibrida berbasis ChromaDB dengan deteksi OOD."""

    DEFAULT_THRESHOLD = 0.50
    DEFAULT_ALPHA = 0.70  # 70% Dense semantic, 30% Lexical keyword

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        threshold: float = DEFAULT_THRESHOLD,
        default_alpha: float = DEFAULT_ALPHA,
    ):
        """Inisialisasi SemanticRetriever.

        Args:
            vector_store: Instansi VectorStore yang digunakan untuk pencarian.
                          Jika None, akan menginisialisasi VectorStore default.
            threshold: Ambang batas minimum skor untuk menerima kueri.
            default_alpha: Bobot dense vector dalam hybrid scoring (0.0 s.d. 1.0).
        """
        self.vector_store = vector_store or VectorStore()
        self.threshold = float(threshold)
        self.default_alpha = float(default_alpha)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
        source_file: Optional[str] = None,
        min_threshold: Optional[float] = None,
        mode: SearchMode = SearchMode.HYBRID,
        alpha: Optional[float] = None,
    ) -> RetrievalResult:
        """Melakukan temu balik hibrida / semantik dengan ambang batas presisi.

        Args:
            query: Teks pertanyaan atau kueri pencarian.
            top_k: Jumlah maksimum potongan dokumen relevan yang dikembalikan.
            doc_type: Filter opsional berdasarkan kategori dokumen.
            source_file: Filter opsional berdasarkan nama berkas PDF sumber.
            min_threshold: Ambang batas spesifik pemanggilan (meng-override default).
            mode: Mode pencarian ("hybrid", "semantic", atau "lexical").
            alpha: Bobot dense vector similarity dalam hybrid search (default 0.70).

        Returns:
            RetrievalResult berisi status kueri, daftar potongan dokumen, dan skor.
        """
        effective_threshold = float(
            min_threshold if min_threshold is not None else self.threshold
        )
        effective_alpha = float(
            alpha if alpha is not None else self.default_alpha
        )
        cleaned_query = query.strip() if query else ""

        if not cleaned_query:
            return RetrievalResult(
                query=query,
                status=QueryStatus.EMPTY_QUERY,
                items=[],
                top_score=0.0,
                threshold=effective_threshold,
                mode=mode,
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

        # Tarik kandidat dokumen dari basis data vektor persisten
        # Mengambil kandidat ekstra untuk re-ranking jika mode hibrida/leksikal
        fetch_limit = min(self.vector_store.count(), max(top_k * 3, 15)) if self.vector_store.count() > 0 else top_k
        raw_results = self.vector_store.query(
            query_text=cleaned_query,
            n_results=fetch_limit,
            where=where_filter,
        )

        if not raw_results:
            return RetrievalResult(
                query=query,
                status=QueryStatus.NO_RESULTS,
                items=[],
                top_score=0.0,
                threshold=effective_threshold,
                mode=mode,
                rejection_reason="Tidak ditemukan dokumen yang cocok dengan kriteria filter.",
            )

        # Hitung skor hibrida / leksikal untuk setiap kandidat
        scored_items: List[SearchResultItem] = []
        for item in raw_results:
            dense_sim = item.similarity_score
            lex_score = compute_lexical_score(cleaned_query, item.text)

            if mode == SearchMode.SEMANTIC:
                final_score = dense_sim
            elif mode == SearchMode.LEXICAL:
                final_score = lex_score
            else:  # SearchMode.HYBRID (Default)
                final_score = round(
                    (effective_alpha * dense_sim) + ((1.0 - effective_alpha) * lex_score), 4
                )

            # Buat instance SearchResultItem baru dengan skor terkalibrasi
            scored_item = SearchResultItem(
                chunk_id=item.chunk_id,
                text=item.text,
                source_file=item.source_file,
                doc_title=item.doc_title,
                doc_type=item.doc_type,
                page_number=item.page_number,
                chunk_index=item.chunk_index,
                similarity_score=final_score,
                distance=round(max(0.0, 1.0 - final_score), 4),
            )
            scored_items.append(scored_item)

        # Urutkan ulang berdasarkan final_score tertinggi
        scored_items.sort(key=lambda x: x.similarity_score, reverse=True)
        top_candidates = scored_items[:top_k]
        top_score = top_candidates[0].similarity_score if top_candidates else 0.0

        # Evaluasi ambang batas kesamaan (Similarity Threshold)
        if top_score < effective_threshold:
            return RetrievalResult(
                query=query,
                status=QueryStatus.REJECTED_OOD,
                items=[],
                top_score=top_score,
                threshold=effective_threshold,
                mode=mode,
                rejection_reason=(
                    f"Kueri berada di luar domain kebencanaan (skor kemiripan tertinggi {top_score:.4f} "
                    f"berada di bawah ambang batas {effective_threshold:.4f})."
                ),
            )

        # Saring hasil agar hanya mengembalikan item yang memenuhi ambang batas
        filtered_items = [
            item for item in top_candidates if item.similarity_score >= effective_threshold
        ]

        return RetrievalResult(
            query=query,
            status=QueryStatus.ACCEPTED,
            items=filtered_items,
            top_score=top_score,
            threshold=effective_threshold,
            mode=mode,
            rejection_reason=None,
        )

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        min_threshold: Optional[float] = None,
        mode: SearchMode = SearchMode.HYBRID,
        alpha: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Melakukan temu balik untuk sekumpulan kueri secara berurutan."""
        return [
            self.retrieve(
                q, top_k=top_k, min_threshold=min_threshold, mode=mode, alpha=alpha
            )
            for q in queries
        ]
