"""Unit test untuk modul chunking terstruktur (src/knowledge_base/chunker.py).

Prinsip TDD (Test-Driven Development):
Memvalidasi word-aware recursive splitting, overlap, rich metadata preservation,
dan penanganan edge cases.
"""

import pytest
from src.knowledge_base.pdf_parser import (
    ParsedDocument,
    DocumentMetadata,
    ExtractedPage,
)
from src.knowledge_base.chunker import (
    TextChunker,
    DocumentChunk,
    ChunkingConfig,
)


class TestChunkingConfig:
    """Pengujian konfigurasi chunking."""

    def test_default_config_valid(self):
        config = ChunkingConfig()
        assert config.chunk_size > 0
        assert config.chunk_overlap >= 0
        assert config.chunk_overlap < config.chunk_size

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap harus lebih kecil"):
            ChunkingConfig(chunk_size=500, chunk_overlap=500)

        with pytest.raises(ValueError, match="chunk_size harus positif"):
            ChunkingConfig(chunk_size=0, chunk_overlap=0)


class TestTextChunker:
    """Pengujian algoritma pemotongan teks terstruktur."""

    @pytest.fixture
    def chunker(self):
        # Gunakan ukuran kecil untuk kemudahan pengujian unit
        config = ChunkingConfig(chunk_size=150, chunk_overlap=30)
        return TextChunker(config=config)

    def test_chunk_text_preserves_words(self, chunker):
        text = (
            "Sistem peringatan dini tsunami BPBD Kota Banda Aceh diaktifkan segera "
            "setelah gempa berpotensi tsunami terdeteksi oleh sensor BMKG. "
            "Masyarakat di pesisir diimbau segera menuju Tempat Evakuasi Sementara (TES). "
            "Jalur evakuasi yang dipilih harus mempertimbangkan tingkat risiko dan kepadatan."
        )
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

        # Pastikan tidak ada kata yang terpotong di tengah huruf
        for c in chunks:
            assert not c.startswith(" ")
            assert not c.endswith(" ")
            # Setiap potongan harus berakhir pada batas kata/tanda baca
            words = c.split()
            assert len(words) >= 3

    def test_chunk_text_creates_overlap(self, chunker):
        text = (
            "Kalimat pertama menjelaskan SOP kesiapsiagaan darurat bencana di wilayah pesisir. "
            "Kalimat kedua merinci rute evakuasi menuju shelter terdekat yang aman. "
            "Kalimat ketiga menegaskan batas waktu emas evakuasi selama 20 menit sebelum gelombang tiba."
        )
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2

        # Cek apakah ada kata yang tumpang tindih antara chunk ke-i dan ke-(i+1)
        overlap_found = False
        words_c0 = set(chunks[0].split())
        words_c1 = set(chunks[1].split())
        shared = words_c0.intersection(words_c1)
        if len(shared) >= 1:
            overlap_found = True

        assert overlap_found, f"Harus ada kata yang overlap antara chunk 0 dan 1. Shared: {shared}"

    def test_chunk_empty_or_short_text(self, chunker):
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   \n\t  ") == []

        short_text = "Hanya satu kalimat pendek."
        res = chunker.chunk_text(short_text)
        assert len(res) == 1
        assert res[0] == short_text

    def test_chunk_parsed_document_preserves_metadata(self, chunker):
        meta = DocumentMetadata(
            file_path="docs/sop_uji.pdf",
            file_name="sop_uji.pdf",
            title="SOP Kesiapsiagaan Bencana",
            doc_type="SOP_Operasional",
            total_pages=2,
            total_words=60,
        )
        page1 = ExtractedPage(
            page_number=1,
            raw_text="Halaman satu berisi pengantar protokol mitigasi tsunami di wilayah pesisir barat.",
            clean_text="Halaman satu berisi pengantar protokol mitigasi tsunami di wilayah pesisir barat.",
            char_count=82,
            word_count=12,
        )
        page2 = ExtractedPage(
            page_number=2,
            raw_text="Halaman dua merinci alur koordinasi Pusdalops saat sirene peringatan dini berbunyi.",
            clean_text="Halaman dua merinci alur koordinasi Pusdalops saat sirene peringatan dini berbunyi.",
            char_count=83,
            word_count=12,
        )
        doc = ParsedDocument(
            metadata=meta,
            pages=[page1, page2],
            full_clean_text=page1.clean_text + "\n\n" + page2.clean_text,
        )

        doc_chunks = chunker.chunk_document(doc)
        assert len(doc_chunks) >= 2

        for i, dc in enumerate(doc_chunks):
            assert isinstance(dc, DocumentChunk)
            assert dc.source_file == "sop_uji.pdf"
            assert dc.doc_title == "SOP Kesiapsiagaan Bencana"
            assert dc.doc_type == "SOP_Operasional"
            assert dc.page_number in (1, 2)
            assert dc.chunk_index == i
            assert dc.chunk_id.startswith("sop_uji_p")
            assert dc.char_count == len(dc.text)
            assert dc.word_count == len(dc.text.split())

