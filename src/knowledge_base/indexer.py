"""Skrip pipeline pengindeksan korpus dokumen ke basis data vektor persisten ChromaDB.

Dijalankan secara langsung dengan:
    uv run python -m src.knowledge_base.indexer

Tanggung jawab utama:
1. Membaca & membersihkan seluruh file PDF di direktori korpus (PDFParser).
2. Membagi dokumen secara rekursif menjadi potongan teks semantik (TextChunker).
3. Mengekstrak dense embedding menggunakan model Sentence-Transformers multibahasa.
4. Menyimpan chunk, vektor, dan rich metadata ke dalam ChromaDB persisten (VectorStore).
"""

import argparse
from pathlib import Path
import time
from typing import Optional

from src.knowledge_base.pdf_parser import PDFParser
from src.knowledge_base.chunker import TextChunker, ChunkingConfig
from src.knowledge_base.embedding_service import SentenceTransformerEmbeddingService
from src.knowledge_base.vector_store import VectorStore


def build_knowledge_base(
    docs_dir: str = "docs/Milestone 3/Dokumen",
    persist_dir: str = "data/chroma_db",
    collection_name: str = "disaster_knowledge_base",
    reset: bool = False,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> VectorStore:
    """Membangun basis data vektor persisten dari direktori dokumen PDF."""
    start_time = time.time()
    docs_path = Path(docs_dir)
    print("=" * 70)
    print("PIPELINE PENGINDEKSAN BASIS DATA VEKTOR CHROMADB")
    print("=" * 70)
    print(f"Direktori Dokumen : {docs_path.resolve()}")
    print(f"Direktori Basis Data: {Path(persist_dir).resolve()}")
    print(f"Koleksi Target     : {collection_name}")
    print(f"Parameter Chunking : Size={chunk_size} char, Overlap={chunk_overlap} char")

    # 1. Ekstraksi Dokumen
    print("\n[1/4] Mengekstrak teks dari seluruh dokumen PDF...")
    parser = PDFParser()
    parsed_docs = parser.parse_directory(docs_path, recursive=True)
    print(f"      -> Berhasil mengekstrak {len(parsed_docs)} dokumen PDF.")

    # 2. Chunking
    print("\n[2/4] Melakukan recursive word-aware chunking...")
    chunker = TextChunker(ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    chunks = chunker.chunk_documents(parsed_docs)
    print(f"      -> Dihasilkan {len(chunks)} potongan teks (chunks) dengan rich metadata.")

    # 3. Inisialisasi Vector Store & Embedding Service
    print("\n[3/4] Menginisialisasi Model Embedding & ChromaDB Persisten...")
    embed_service = SentenceTransformerEmbeddingService(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        batch_size=64,
    )
    vector_store = VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_service=embed_service,
    )

    if reset:
        print("      -> Mereset koleksi sebelumnya sesuai permintaan flag --reset.")
        vector_store.reset()

    initial_count = vector_store.count()
    print(f"      -> Jumlah item tersimpan sebelum indexing: {initial_count}")

    # 4. Batch Indexing
    print("\n[4/4] Menyimpan dense embeddings dan metadata ke ChromaDB...")
    inserted = vector_store.add_chunks(chunks, batch_size=128)
    final_count = vector_store.count()
    duration = time.time() - start_time

    print("\n" + "=" * 70)
    print("RINGKASAN HASIL INDEXING")
    print("=" * 70)
    print(f"Total Dokumen Diproses : {len(parsed_docs)}")
    print(f"Total Chunks Tersimpan  : {final_count} (Baru ditambah: {inserted})")
    print(f"Dimensi Vektor Embedding: {embed_service.dimension} dimensi")
    print(f"Waktu Eksekusi Total   : {duration:.2f} detik")
    print("=" * 70)

    # Uji verifikasi temu balik
    sample_query = "Bagaimana prosedur aktivasi sirine peringatan dini tsunami?"
    print(f"\n[Verifikasi] Menjalankan uji temu balik contoh:")
    print(f"Query: '{sample_query}'")
    hits = vector_store.query(sample_query, n_results=2)
    for idx, hit in enumerate(hits, start=1):
        print(f"\n  Result #{idx} (Similarity: {hit.similarity_score:.4f} | Distance: {hit.distance:.4f}):")
        print(f"  - Dokumen : {hit.source_file} (Hal. {hit.page_number})")
        print(f"  - Kategori: {hit.doc_type}")
        print(f"  - Cuplikan: {hit.text[:150]}...")

    return vector_store


def main():
    parser = argparse.ArgumentParser(description="Indexer Dokumen PDF ke ChromaDB")
    parser.add_argument("--docs-dir", default="docs/Milestone 3/Dokumen", help="Path direktori file PDF")
    parser.add_argument("--persist-dir", default="data/chroma_db", help="Path direktori persisten ChromaDB")
    parser.add_argument("--collection-name", default="disaster_knowledge_base", help="Nama koleksi ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Reset koleksi sebelum indexing")
    args = parser.parse_args()

    build_knowledge_base(
        docs_dir=args.docs_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()

