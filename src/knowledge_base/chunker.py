"""Modul pemotongan teks terstruktur (Word-Aware Recursive Chunking).

Mengimplementasikan:
- Pemotongan hierarkis (paragraf -> kalimat -> kata).
- Overlap terukur untuk menjaga kontinuitas semantik pada batas kalimat.
- Penyematan rich metadata (nama dokumen, kategori, nomor halaman, slug ID).
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import List, Tuple
from src.knowledge_base.pdf_parser import ParsedDocument


@dataclass(frozen=True)
class ChunkingConfig:
    """Konfigurasi parameter chunking dengan justifikasi ilmiah."""

    # ~700 karakter (~100-120 kata): optimal untuk model embedding Sentence-Transformers (max 256 token)
    chunk_size: int = 700
    # ~120 karakter (~17% overlap): mempertahankan konteks klausa/kalimat pada perbatasan chunk
    chunk_overlap: int = 120
    # Hierarki pemisah dari level terstruktur ke level kata
    separators: Tuple[str, ...] = field(
        default=(
            "\n\n",  # Paragraf
            "\n",    # Baris
            ". ",    # Kalimat berita
            "? ",    # Kalimat tanya
            "! ",    # Kalimat seru
            "; ",    # Klausa regulasi/SOP
            " ",     # Kata (fallback terendah)
        )
    )

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size harus positif, dapat: {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap tidak boleh negatif, dapat: {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap harus lebih kecil dari chunk_size ({self.chunk_overlap} >= {self.chunk_size})"
            )


@dataclass(frozen=True)
class DocumentChunk:
    """Potongan teks semantik yang siap di-embedding dan disimpan ke ChromaDB."""

    chunk_id: str
    text: str
    source_file: str
    doc_title: str
    doc_type: str
    page_number: int
    chunk_index: int
    char_count: int
    word_count: int


def _slugify(text: str) -> str:
    """Membuat slug ringkas dari nama file untuk ID chunk yang bersih."""
    stem = Path(text).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem)
    return slug.strip("_")[:30]


class TextChunker:
    """Pemotong teks dokumen berbasis aturan hierarkis kata dan kalimat."""

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()

    def chunk_text(self, text: str) -> List[str]:
        """Membagi teks menjadi daftar potongan teks dengan ukuran dan overlap terukur."""
        if not text or not text.strip():
            return []

        clean = text.strip()
        if len(clean) <= self.config.chunk_size:
            return [clean]

        return self._recursive_split(clean, self.config.separators)

    def _recursive_split(self, text: str, separators: Tuple[str, ...]) -> List[str]:
        """Secara rekursif membagi teks menggunakan hierarki pemisah."""
        if not text:
            return []

        if len(text) <= self.config.chunk_size:
            return [text.strip()]

        # Cari pemisah yang ada dalam teks
        chosen_sep = None
        remaining_seps = ()
        for idx, sep in enumerate(separators):
            if sep in text:
                chosen_sep = sep
                remaining_seps = separators[idx + 1 :]
                break

        # Jika tidak ada pemisah yang cocok sama sekali, potong paksa pada spasi
        if chosen_sep is None:
            words = text.split()
            chunks = []
            curr_words = []
            curr_len = 0
            for w in words:
                w_len = len(w) + 1
                if curr_len + w_len > self.config.chunk_size and curr_words:
                    chunks.append(" ".join(curr_words))
                    curr_words = [w]
                    curr_len = len(w)
                else:
                    curr_words.append(w)
                    curr_len += w_len
            if curr_words:
                chunks.append(" ".join(curr_words))
            return chunks

        # Pisahkan dengan chosen_sep
        splits = text.split(chosen_sep)
        chunks: List[str] = []
        current_doc: List[str] = []
        current_len = 0

        for s in splits:
            s_len = len(s) + (len(chosen_sep) if current_doc else 0)

            # Jika potongan individual melebihi chunk_size, rekursif dengan separator berikutnya
            if len(s) > self.config.chunk_size:
                # Simpan chunk saat ini jika ada
                if current_doc:
                    full_chunk = chosen_sep.join(current_doc).strip()
                    if full_chunk:
                        chunks.append(full_chunk)
                    current_doc = []
                    current_len = 0

                # Rekursif untuk s yang besar
                if remaining_seps:
                    sub_chunks = self._recursive_split(s, remaining_seps)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(s.strip())
                continue

            if current_len + s_len > self.config.chunk_size:
                full_chunk = chosen_sep.join(current_doc).strip()
                if full_chunk:
                    chunks.append(full_chunk)

                # Siapkan overlap: ambil elemen terakhir dari current_doc
                overlap_doc: List[str] = []
                overlap_len = 0
                for item in reversed(current_doc):
                    if overlap_len + len(item) <= self.config.chunk_overlap:
                        overlap_doc.insert(0, item)
                        overlap_len += len(item) + len(chosen_sep)
                    else:
                        break

                current_doc = overlap_doc + [s]
                current_len = sum(len(x) for x in current_doc) + len(chosen_sep) * max(0, len(current_doc) - 1)
            else:
                current_doc.append(s)
                current_len += s_len

        if current_doc:
            final_chunk = chosen_sep.join(current_doc).strip()
            if final_chunk:
                chunks.append(final_chunk)

        return chunks

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """Membagi seluruh isi satu dokumen PDF menjadi DocumentChunk dengan rich metadata."""
        document_chunks: List[DocumentChunk] = []
        slug = _slugify(doc.metadata.file_name)
        global_idx = 0

        for page in doc.pages:
            if not page.clean_text:
                continue

            text_chunks = self.chunk_text(page.clean_text)
            for page_chunk_idx, text_part in enumerate(text_chunks):
                chunk_id = f"{slug}_p{page.page_number}_c{page_chunk_idx}"
                chunk_obj = DocumentChunk(
                    chunk_id=chunk_id,
                    text=text_part,
                    source_file=doc.metadata.file_name,
                    doc_title=doc.metadata.title,
                    doc_type=doc.metadata.doc_type,
                    page_number=page.page_number,
                    chunk_index=global_idx,
                    char_count=len(text_part),
                    word_count=len(text_part.split()),
                )
                document_chunks.append(chunk_obj)
                global_idx += 1

        return document_chunks

    def chunk_documents(self, docs: List[ParsedDocument]) -> List[DocumentChunk]:
        """Membagi sekumpulan dokumen PDF menjadi daftar DocumentChunk yang terpadu."""
        all_chunks: List[DocumentChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

