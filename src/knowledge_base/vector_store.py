"""Modul penyimpanan vektor persisten menggunakan ChromaDB.

Menyediakan:
- Inisialisasi basis data persisten (PersistentClient).
- Penyimpanan DocumentChunk dengan dense embedding dan rich metadata.
- Temu balik semantik (Cosine Similarity) dan penyaringan metadata (metadata filtering).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from src.knowledge_base.chunker import DocumentChunk
from src.knowledge_base.embedding_service import (
    BaseEmbeddingService,
    SentenceTransformerEmbeddingService,
)


@dataclass(frozen=True)
class SearchResultItem:
    """Hasil temu balik satu dokumen dari basis data vektor."""

    chunk_id: str
    text: str
    source_file: str
    doc_title: str
    doc_type: str
    page_number: int
    chunk_index: int
    similarity_score: float
    distance: float


class VectorStore:
    """Pengelola koleksi vektor persisten ChromaDB."""

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "disaster_knowledge_base",
        embedding_service: Optional[BaseEmbeddingService] = None,
    ):
        project_root = Path(__file__).resolve().parent.parent.parent
        target_path = Path(persist_dir)
        if not target_path.is_absolute():
            if (Path.cwd() / target_path).exists():
                self.persist_dir = str((Path.cwd() / target_path).resolve())
            else:
                self.persist_dir = str((project_root / target_path).resolve())
        else:
            self.persist_dir = str(target_path.resolve())

        self.collection_name = collection_name
        self.embedding_service = embedding_service or SentenceTransformerEmbeddingService()

        # Inisialisasi klien persisten ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Buat atau ambil koleksi dengan metrik Cosine Distance
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self, chunks: List[DocumentChunk], batch_size: int = 128
    ) -> int:
        """Menyimpan daftar DocumentChunk beserta vektor dense-nya ke ChromaDB."""
        if not chunks:
            return 0

        total_inserted = 0
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [
                {
                    "source_file": c.source_file,
                    "doc_title": c.doc_title,
                    "doc_type": c.doc_type,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "char_count": c.char_count,
                    "word_count": c.word_count,
                }
                for c in batch
            ]

            embeddings = self.embedding_service.embed_texts(texts)

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_inserted += len(batch)

        return total_inserted

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResultItem]:
        """Melakukan pencarian semantik berbasis Cosine Similarity."""
        if self.count() == 0 or not query_text.strip():
            return []

        query_vector = self.embedding_service.embed_query(query_text)

        # Batasi n_results agar tidak melebihi jumlah item di collection
        k = min(n_results, self.count())

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": k,
        }
        if where:
            kwargs["where"] = where

        raw_results = self.collection.query(**kwargs)

        items: List[SearchResultItem] = []
        if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
            return []

        ids = raw_results["ids"][0]
        docs = raw_results["documents"][0] if raw_results.get("documents") else []
        metas = raw_results["metadatas"][0] if raw_results.get("metadatas") else []
        distances = raw_results["distances"][0] if raw_results.get("distances") else []

        for idx, chunk_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            dist = float(distances[idx]) if idx < len(distances) else 1.0
            # Pada hnsw:space cosine di ChromaDB: distance = 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - dist))

            item = SearchResultItem(
                chunk_id=chunk_id,
                text=docs[idx] if idx < len(docs) else "",
                source_file=meta.get("source_file", ""),
                doc_title=meta.get("doc_title", ""),
                doc_type=meta.get("doc_type", ""),
                page_number=int(meta.get("page_number", 1)),
                chunk_index=int(meta.get("chunk_index", 0)),
                similarity_score=round(similarity, 4),
                distance=round(dist, 4),
            )
            items.append(item)

        return items

    def count(self) -> int:
        """Mengembalikan jumlah dokumen chunk yang tersimpan di koleksi."""
        return self.collection.count()

    def reset(self) -> None:
        """Mengosongkan seluruh koleksi saat ini."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

