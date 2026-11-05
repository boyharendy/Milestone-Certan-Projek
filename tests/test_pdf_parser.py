"""Unit test untuk modul ekstraksi dan pembersihan teks PDF (src/knowledge_base/pdf_parser.py).

Mengikuti prinsip TDD (Test-Driven Development).
"""

import os
from pathlib import Path
import pytest
from pypdf import PdfWriter

# Impor fungsi dan kelas dari modul yang akan diimplementasikan
from src.knowledge_base.pdf_parser import (
    clean_text,
    infer_doc_type,
    infer_doc_title,
    PDFParser,
    ExtractedPage,
    DocumentMetadata,
    ParsedDocument,
)


class TestTextCleaner:
    """Pengujian untuk fungsi pembersihan teks (clean_text)."""

    def test_clean_text_normalizes_whitespace(self):
        raw = "Ini   adalah   contoh   teks \t dengan  spasi \n\n\n berlebih."
        cleaned = clean_text(raw)
        assert "   " not in cleaned
        assert cleaned == "Ini adalah contoh teks dengan spasi\n\nberlebih."

    def test_clean_text_fixes_hyphenated_linebreaks(self):
        raw = "Pemerintah melakukan pe-\nrencanaan tanggap darurat ke-\nbencanaan."
        cleaned = clean_text(raw)
        assert "perencanaan" in cleaned
        assert "kebencanaan" in cleaned
        assert "pe-\nrencanaan" not in cleaned

    def test_clean_text_removes_unprintable_chars(self):
        raw = "Teks dengan karakter anomali \x00\x08\x0b\x0c dan simbol valid."
        cleaned = clean_text(raw)
        assert "\x00" not in cleaned
        assert "\x08" not in cleaned
        assert "Teks dengan karakter anomali dan simbol valid." in cleaned

    def test_clean_text_empty_and_whitespace(self):
        assert clean_text("") == ""
        assert clean_text("   \n\t  ") == ""
        assert clean_text(None) == ""


class TestMetadataInference:
    """Pengujian untuk inferensi metadata dokumen dari nama file."""

    def test_infer_doc_type(self):
        assert infer_doc_type("a02_buku_2_provinsi_aceh_sops1.pdf") == "SOP_Operasional"
        assert infer_doc_type("Pergub Provinsi Aceh Nomor 54 Tahun 2023.pdf") == "Regulasi_Hukum"
        assert infer_doc_type("Qanun_Aceh_Nomor_5_Tahun_2010_Tentang_Pe.pdf") == "Regulasi_Hukum"
        assert infer_doc_type("In_Pres No.1 Tahun 2005.pdf") == "Regulasi_Hukum"
        assert infer_doc_type("perka-15-tahun-2012-tentang-pedoman.pdf") == "Regulasi_Hukum"
        assert infer_doc_type("Perencanaan Tempat Evakuasi Sementara Sunami.pdf") == "Pedoman_Teknis"
        assert infer_doc_type("Pedoman_Sosialisasi_Penanggulangan_Bencana.pdf") == "Pedoman_Teknis"
        assert infer_doc_type("rencana_kerja_bpba_thn_20241.pdf") == "Rencana_Kerja"
        assert infer_doc_type("buku_aceh_pasca_lima_belas_tahun_tsunami1.pdf") == "Kajian_Akademik"
        assert infer_doc_type("meydinafa,+Journal+manager,+JP+V10+N1_37-48.pdf") == "Kajian_Akademik"
        assert infer_doc_type("dokumen_acak.pdf") == "Dokumen_Umum"

    def test_infer_doc_title(self):
        title = infer_doc_title("Pergub Provinsi Aceh Nomor 54 Tahun 2023.pdf")
        assert "Pergub Provinsi Aceh Nomor 54 Tahun 2023" in title
        assert not title.endswith(".pdf")


class TestPDFParser:
    """Pengujian fungsionalitas PDFParser."""

    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Membuat file PDF minimal valid untuk pengujian."""
        pdf_path = tmp_path / "sample_sop.pdf"
        writer = PdfWriter()
        # Tambahkan 2 halaman kosong
        writer.add_blank_page(width=200, height=200)
        writer.add_blank_page(width=200, height=200)
        with open(pdf_path, "wb") as f:
            writer.write(f)
        return str(pdf_path)

    def test_parse_file_success(self, sample_pdf_path):
        parser = PDFParser()
        doc = parser.parse_file(sample_pdf_path)

        assert isinstance(doc, ParsedDocument)
        assert doc.metadata.file_name == "sample_sop.pdf"
        assert doc.metadata.doc_type == "SOP_Operasional"
        assert doc.metadata.total_pages == 2
        assert len(doc.pages) == 2
        assert doc.pages[0].page_number == 1
        assert doc.pages[1].page_number == 2

    def test_parse_file_not_found_raises(self):
        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("non_existent_file.pdf")

    def test_parse_non_pdf_raises(self, tmp_path):
        txt_path = tmp_path / "notes.txt"
        txt_path.write_text("bukan pdf", encoding="utf-8")
        parser = PDFParser()
        with pytest.raises(ValueError, match="Harus berupa file PDF"):
            parser.parse_file(str(txt_path))

    def test_parse_directory(self, tmp_path):
        # Buat 2 PDF dan 1 TXT
        p1 = tmp_path / "doc1_qanun.pdf"
        p2 = tmp_path / "doc2_sop.pdf"
        p3 = tmp_path / "ignore.txt"

        for p in [p1, p2]:
            w = PdfWriter()
            w.add_blank_page(width=100, height=100)
            with open(p, "wb") as f:
                w.write(f)
        p3.write_text("abaikan", encoding="utf-8")

        parser = PDFParser()
        docs = parser.parse_directory(str(tmp_path))
        assert len(docs) == 2
        names = {d.metadata.file_name for d in docs}
        assert names == {"doc1_qanun.pdf", "doc2_sop.pdf"}

