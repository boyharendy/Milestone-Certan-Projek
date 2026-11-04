"""Modul ekstraksi teks dan pembersihan dokumen PDF kebencanaan.

Menyediakan pipeline pemrosesan teks dokumen:
- Ekstraksi teks per halaman dengan pypdf.
- Pembersihan noise (whitespace, hyphenated line break, unprintable chars).
- Inferensi metadata (tipe dokumen, judul, jumlah halaman, statistik kata).
"""

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import List, Optional
from pypdf import PdfReader


@dataclass(frozen=True)
class ExtractedPage:
    """Representasi data satu halaman PDF yang telah diekstrak dan dibersihkan."""

    page_number: int
    raw_text: str
    clean_text: str
    char_count: int
    word_count: int


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata struktural dokumen PDF."""

    file_path: str
    file_name: str
    title: str
    doc_type: str
    total_pages: int
    total_words: int


@dataclass(frozen=True)
class ParsedDocument:
    """Hasil akhir pemrosesan dokumen PDF utuh."""

    metadata: DocumentMetadata
    pages: List[ExtractedPage]
    full_clean_text: str


def clean_text(text: Optional[str]) -> str:
    """Membersihkan teks mentah hasil ekstraksi PDF.

    Langkah pembersihan:
    1. Menangani input None/kosong.
    2. Normalisasi Unicode (NFKC).
    3. Menghapus karakter unprintable dan control characters (kecuali newline & tab).
    4. Menyambung kata yang terpotong tanda hubung di akhir baris (hyphenation).
    5. Menghilangkan spasi dan tab berlebih pada tiap baris.
    6. Menjaga batas paragraf (maksimal dua baris baru berurutan).
    """
    if not text:
        return ""

    # 1. Normalisasi Unicode
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Hapus control characters yang anomali (ASCII 0-8, 11-12, 14-31)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)

    # 3. Standardisasi newline ke format Unix (\n)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Perbaiki kata bersambung akibat pemenggalan baris: pe-\nrencanaan -> perencanaan
    cleaned = re.sub(r"(\b\w+)-\s*\n\s*(\w+\b)", r"\1\2", cleaned)

    # 5. Bersihkan spasi/tab horizontal ganda per baris
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]

    # 6. Gabungkan kembali baris, membatasi maksimal 2 newline berturut-turut
    merged = "\n".join(lines)
    merged = re.sub(r"\n{3,}", "\n\n", merged)

    return merged.strip()


def infer_doc_type(file_name: str) -> str:
    """Mengidentifikasi tipe dokumen berdasarkan nama file."""
    lower_name = file_name.lower()

    if "sop" in lower_name:
        return "SOP_Operasional"
    if any(k in lower_name for k in ["pergub", "qanun", "perka"]):
        return "Regulasi_Hukum"
    if any(k in lower_name for k in ["pedoman", "perencanaan"]):
        return "Pedoman_Teknis"
    if "rencana_kerja" in lower_name:
        return "Rencana_Kerja"
    if any(k in lower_name for k in ["buku", "journal", "meydinafa"]):
        return "Kajian_Akademik"

    return "Dokumen_Umum"


def infer_doc_title(file_name: str) -> str:
    """Menghasilkan judul dokumen yang rapi dari nama file."""
    base = Path(file_name).stem
    # Ganti underscore dan tanda plus dengan spasi
    cleaned = re.sub(r"[_+]+", " ", base)
    # Rapikan spasi ganda
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class PDFParser:
    """Ekstraktor dan pembersih dokumen PDF untuk basis pengetahuan."""

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """Mengekstrak dan memproses sebuah file PDF."""
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File PDF tidak ditemukan: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Harus berupa file PDF: {path}")

        reader = PdfReader(str(path))
        extracted_pages: List[ExtractedPage] = []
        full_text_parts: List[str] = []
        total_words = 0

        for idx, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            clean = clean_text(raw)
            words = len(clean.split()) if clean else 0
            total_words += words

            page_obj = ExtractedPage(
                page_number=idx + 1,
                raw_text=raw,
                clean_text=clean,
                char_count=len(clean),
                word_count=words,
            )
            extracted_pages.append(page_obj)

            if clean:
                full_text_parts.append(clean)

        full_text = "\n\n".join(full_text_parts)
        metadata = DocumentMetadata(
            file_path=str(path),
            file_name=path.name,
            title=infer_doc_title(path.name),
            doc_type=infer_doc_type(path.name),
            total_pages=len(extracted_pages),
            total_words=total_words,
        )

        return ParsedDocument(
            metadata=metadata,
            pages=extracted_pages,
            full_clean_text=full_text,
        )

    def parse_directory(
        self, dir_path: str | Path, recursive: bool = False
    ) -> List[ParsedDocument]:
        """Mengekstrak seluruh file PDF dalam suatu direktori."""
        target_dir = Path(dir_path).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Direktori tidak ditemukan: {target_dir}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(list(target_dir.glob(pattern)))

        parsed_docs: List[ParsedDocument] = []
        for pdf_file in pdf_files:
            parsed = self.parse_file(pdf_file)
            parsed_docs.append(parsed)

        return parsed_docs

