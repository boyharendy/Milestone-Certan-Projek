# Perancangan Sistem Rekomendasi Rute dan Basis Pengetahuan Kebencanaan Berbasis Agen Cerdas di BPBD Kota Banda Aceh

Repositori ini mengembangkan prototipe sistem pendukung keputusan mitigasi dan evakuasi tsunami untuk BPBD Kota Banda Aceh secara bertahap:
- **Milestone 1:** Memodelkan jaringan evakuasi sebagai graf dan membandingkan *Uniform Cost Search* (UCS) dengan *A\** Search.
- **Milestone 2:** Mengimplementasikan *Genetic Algorithm* (GA) untuk mengoptimasi alokasi titik bahaya kepada armada evakuasi dan shelter berdasarkan matriks waktu tempuh OSMnx.
- **Milestone 3:** Membangun memori semantik perusahaan (*Enterprise Knowledge Base*) menggunakan basis data vektor persisten **ChromaDB**, model **Sentence-Transformers** multilingual, dan **Semantic Retriever** cerdas berbasis ambang batas (*confidence threshold*) untuk menolak kueri di luar domain (*out-of-domain*).

---

## Tim Pengembang (Kelompok 18)

| NIM | Nama | Peran & Tanggung Jawab Utama |
| --- | --- | --- |
| **12S24010** | **Artha Liebe Siregar** | **Data & Ingestion Engineer:** Kurasi korpus enterprise, ekstraksi teks PDF, pembersihan noise, dan implementasi *word-aware recursive chunking*. |
| **12S24016** | **Boy Harendy Simamora** | **Vector Store & ML Pipeline Engineer:** Pemodelan graf, algoritma GA, seleksi model Sentence-Transformers, arsitektur ChromaDB persisten, dan batch indexing. |
| **12S24044** | **Dianita Lorensia Br Ginting** | **Semantic Evaluation & QA Specialist:** Pengujian unit TDD, perancangan kueri bisnis, evaluasi ambang batas similarity OOD, benchmark temu balik, dan dokumentasi. |

---

## Latar Belakang (Background)

Kota Banda Aceh memiliki wilayah pesisir dengan risiko tinggi terhadap bencana gempabumi tektonik dan gelombang tsunami. Dalam situasi kedaruratan maupun fase mitigasi bencana:
1. **Navigasi Spasial:** Keputusan rute evakuasi harus mempertimbangkan jarak, risiko jalur, kemacetan, serta batas kapasitas shelter.
2. **Akses Regulasi Enterprise:** Dokumen SOP Pusdalops PB, qanun daerah, instruksi presiden, dan pedoman teknis tempat evakuasi tersebar dalam format PDF statis yang tebal dan rumit. Sistem pencarian berbasis teks konvensional (seperti regex atau BM25) gagal menangani kueri non-literal, parafrase bahasa awam, dan kueri luar domain.

Proyek ini dibangun sebagai prototipe terpadu agen cerdas kebencanaan yang menggabungkan kemampuan **optimasi rute graf** dan **temu balik memori semantik korpus (*semantic search knowledge base*)** menuju arsitektur *Retrieval-Augmented Generation* (RAG).

---

## Struktur Proyek

```text
Milestone/
├── docs/
│   ├── Milestone 1/
│   │   ├── peas_specification.md       # Matriks PEAS dan karakteristik lingkungan
│   │   └── problem_framing.md          # Analisis masalah bisnis BPBD
│   ├── Milestone 2/                    # Naskah dan visualisasi alokasi GA
│   └── Milestone 3/
│       └── Dokumen/                    # 10 Dokumen PDF resmi korpus kebencanaan
├── data/
│   ├── distance_matrix.json            # Matriks waktu tempuh titik bahaya ke shelter
│   ├── banda_aceh_drive.meta.json      # Metadata graf jalan Banda Aceh
│   └── chroma_db/                      # Basis data vektor persisten ChromaDB (1.660 chunks)
├── output/
│   ├── benchmark_results.json          # Hasil eksperimen benchmark GA Milestone 2
│   ├── benchmark_runtime.png           # Grafik runtime GA
│   ├── benchmark_convergence.png       # Grafik konvergensi fitness GA
│   ├── peta_jaringan_evakuasi.png      # Visualisasi graf evakuasi Milestone 1
│   ├── peta_rute_riil.png              # Plotting rute OpenStreetMap
│   ├── retrieval_benchmark.json        # Hasil evaluasi 12 kueri bisnis Milestone 3
│   └── retrieval_similarity.png        # Grafik analisis ambang batas similarity OOD
├── src/
│   ├── knowledge_base/                 # Modul Utama Milestone 3
│   │   ├── __init__.py                 # Ekspor komponen basis pengetahuan
│   │   ├── pdf_parser.py               # Ekstraksi PDF, normalisasi NFKC, & metadata
│   │   ├── chunker.py                  # Word-aware recursive character chunking
│   │   ├── embedding_service.py        # Sentence-Transformers MiniLM (384-dim)
│   │   ├── vector_store.py             # ChromaDB PersistentClient & path resolution
│   │   ├── indexer.py                  # Batch indexing pipeline (1.660 chunks)
│   │   └── retriever.py                # Semantic retriever & confidence thresholding
│   ├── benchmark_retrieval.py          # Benchmark temu balik 12 kueri uji bisnis
│   ├── data_loader.py                  # Data loader OSMnx dan matriks jarak
│   ├── ga_solver.py                    # Solver Genetic Algorithm untuk alokasi armada
│   ├── graph_model.py                  # Representasi graf evakuasi Banda Aceh
│   ├── heuristics.py                   # Heuristik Euclidean untuk A*
│   ├── search.py                       # Algoritma pencarian UCS dan A*
│   ├── visualize.py                    # Visualisasi kinerja graf Milestone 1
│   └── visualize_map.py                # Visualisasi rute evakuasi peta riil
├── tests/                              # Suite Pengujian Otomatis (70 Unit Tests)
│   ├── test_pdf_parser.py              # Pengujian parser teks, normalisasi, & pemindaian rekursif
│   ├── test_chunker.py                 # Pengujian recursive chunking & overlap
│   ├── test_vector_store.py            # Pengujian ChromaDB persisten & hybrid filter
│   ├── test_retrieval.py               # Pengujian True Hybrid Search, skor leksikal, & OOD thresholding
│   ├── test_benchmark_retrieval.py     # Pengujian struktur kueri bisnis, ground truth, & plot
│   ├── test_data_loader.py             # Pengujian matriks jarak dan peta
│   ├── test_search.py                  # Pengujian model graf dan optimalitas UCS/A*
│   └── test_solver.py                  # Pengujian operator dan penalti GA
├── pyproject.toml                      # Konfigurasi proyek dan dependensi uv
├── uv.lock                             # Versi dependensi terkunci
└── README.md
```

---

## Milestone 3: Basis Pengetahuan Korpus & Vector Semantic Search

### 1. Kurasi Korpus Otoritatif (10 Dokumen Enterprise)
Korpus mencakup 10 dokumen otoritatif kebencanaan Aceh yang terbagi ke dalam empat kategori:
1. `In_Pres No.1 Tahun 2005.pdf` (*Regulasi Hukum* - Penanganan Darurat & Rekonstruksi Pasca Tsunami NAD)
2. `Pergub Provinsi Aceh Nomor 54 Tahun 2023.pdf` (*Regulasi Hukum* - Struktur Penyelenggaraan PB)
3. `Qanun_Aceh_Nomor_5_Tahun_2010_Tentang_Pe.pdf` (*Regulasi Hukum* - Pokok Penanggulangan Bencana)
4. `perka-15-tahun-2012 Pusdalops PB.pdf` (*Regulasi Hukum* - Pedoman Operasi Pusdalops)
5. `perka-2-tahun-2012 Pedoman Risiko Bencana.pdf` (*Regulasi Hukum* - Kajian Risiko Bencana)
6. `a02_buku_2_provinsi_aceh_sops1.pdf` (*SOP Operasional* - SOP Kesiapsiagaan Daerah)
7. `Pedoman_Sosialisasi_Penanggulangan_Bencana.pdf` (*Pedoman Teknis* - Prosedur Evakuasi Warga)
8. `Perencanaan Tempat Evakuasi Sementara Sunami.pdf` (*Pedoman Teknis* - Standar Bangunan TES)
9. `buku_aceh_pasca_lima_belas_tahun_tsunami1.pdf` (*Kajian Akademik* - Evaluasi 15 Tahun Tsunami)
10. `meydinafa,+Journal+manager,+JP+V10+N1_37-48.pdf` (*Kajian Akademik* - Tata Ruang Wilayah Pesisir)

### 2. Prapemrosesan & Pembersihan Teks (`pdf_parser.py`, `indexer.py`)
- Ekstraksi teks per halaman menggunakan pustaka `pypdf`.
- **Pemindaian Direktori Rekursif:** `indexer.py` memanggil parser dengan `recursive=True` sehingga menjangkau seluruh hierarki folder dokumen tanpa ada berkas yang terlewat.
- **Normalisasi NFKC** untuk standarisasi ligatur dan karakter tipografi Unicode.
- **Hyphenation Repair:** Menyambung kata yang terpotong tanda hubung di akhir baris (`r'(\w+)-\n(\w+)' -> r'\1\2'`).
- **Sanitasi Whitespace:** Menghilangkan karakter kontrol non-printable dan merapikan spasi berulang.

### 3. Word-Aware Recursive Chunking (`chunker.py`)
- **Algoritma:** Memprioritaskan batas alami pemisah paragraf (`\n\n`), baris baru (`\n`), tanda baca kalimat, dan spasi antar-kata tanpa memotong di tengah kata.
- **Justifikasi Ilmiah Parameter:**
  - *Chunk Size = 700 karakter (~100–120 kata):* Sesuai literatur RAG/Sentence-Transformers (Reimers & Gurevych, 2019) sebagai ukuran ideal representasi semantik klausul regulasi tanpa dilusi informasi.
  - *Chunk Overlap = 120 karakter (~17%):* Menjamin tidak ada konteks penting pada batas antarkalimat yang terputus.
- **Hasil:** Menghasilkan **1.660 chunks** terindeks lengkap dengan *rich metadata*.

### 4. Pipeline Dense Embedding & ChromaDB (`embedding_service.py`, `vector_store.py`)
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (output 384 dimensi, normalisasi L2).
- **Persistent Storage:** `chromadb.PersistentClient` pada direktori `data/chroma_db/`, koleksi `disaster_knowledge_base`, metrik jarak `cosine` ($Distance = 1 - CosineSimilarity$).
- **Resolusi Jalur Absolut:** Menggunakan pendeteksi project root otomatis (`Path(__file__).resolve().parent.parent.parent / "data/chroma_db"`) sehingga sistem berjalan konsisten dari direktori mana pun.

### 5. True Hybrid Search & Confidence Thresholding (`retriever.py`)
Retriever menerapkan arsitektur **True Hybrid Search** menggabungkan dua paradigma temu balik informasi:
1. **Dense Semantic Search:** Menangkap kedekatan semantik, hubungan konseptual, dan parafrase bahasa alami berbasis representasi vektor Sentence-Transformers ($Score_{dense}$).
2. **Sparse / Lexical Keyword Search:** Mengukur kecocokan istilah teknis, nama regulasi, dan akronim melalui representasi overlap token dan frekuensi term ($Score_{lexical}$).
3. **Hybrid Scoring Formula:**
   $$Score_{hybrid} = \alpha \cdot Score_{dense} + (1 - \alpha) \cdot Score_{lexical}$$
   dengan bobot seimbang terkalibrasi $\alpha = 0.70$ (70% semantik, 30% leksikal).
4. **Mode Pencarian Fleksibel:** Mendukung `SearchMode.HYBRID` (default), `SearchMode.SEMANTIC` (vektor murni), dan `SearchMode.LEXICAL` (leksikal murni).
5. **Confidence Thresholding & OOD Rejection:**
   - Ambang batas $\tau = 0.50$ diterapkan setelah re-ranking hibrida.
   - Jika $\text{Top Score} \ge 0.50 \implies$ Kueri diterima (`ACCEPTED`) beserta potongan dokumen relevan.
   - Jika $\text{Top Score} < 0.50 \implies$ Kueri ditolak secara presisi (`REJECTED_OOD`) sebagai kueri di luar domain kebencanaan.
   - Mendukung penyaringan metadata bertingkat (`doc_type` dan `source_file`).

### 6. Hasil Benchmark Temu Balik Semantik (12 Kueri Bisnis)

Pengujian benchmark dieksekusi melalui skrip `src/benchmark_retrieval.py` dengan evaluasi status kueri dan *ground truth target documents*:

| Kategori Kueri | Jumlah Kueri | Rentang Similarity | Status Evaluasi | Relevansi Ground Truth (Top-1) |
| :--- | :---: | :---: | :---: | :---: |
| **Spesifik / Regulasi (SOP)** | 4 | **0.7245 – 0.8698** | `accepted` | 100% (4/4 Cocok Dokumen Resmi) |
| **Parafrase Bahasa Awam** | 2 | **0.5745 – 0.6994** | `accepted` | 100% (2/2 Cocok Dokumen Resmi) |
| **Ambigu / Singkat** | 2 | **0.5138 – 0.8555** | `accepted` | 100% (2/2 Cocok Dokumen Resmi) |
| **Out-of-Domain (OOD)** | 4 | **0.2232 – 0.3930** | `rejected_out_of_domain` | 100% (4/4 Ditolak Tanpa Halusinasi) |

#### Ringkasan Metrik Kinerja:
- **Akurasi Status (In/OOD):** **100.00% (12 / 12 kueri berhasil diklasifikasi tepat)**
- **Relevansi Ground Truth (Top-1 & Top-K):** **100.00% (12 / 12 kueri menghasilkan dokumen rujukan otoritatif yang benar)**
- **In-Domain Recall:** **100.00% (8 / 8 kueri kebencanaan diterima)**
- **OOD Rejection Precision:** **100.00% (4 / 4 kueri luar domain ditolak)**
- **Rata-rata Kemiripan In-Domain:** **0.7188**
- **Rata-rata Kemiripan OOD:** **0.3118**
- **Margin Pemisahan (Separation Gap):** **0.4071** (~40.71% margin keamanan pemisahan)
- **Karakteristik Latensi Sistem:**
  - *Inisialisasi Model Awal (Cold-Start):* ~50 detik (waktu satu kali pemuatan bobot model PyTorch & Sentence-Transformers ke CPU Windows saat proses booting).
  - *Latensi Inferensi Operasional (Steady-State):* **48.82 ms** per kueri (diukur transparan pasca warm-up kueri, mencakup pencarian vektor ChromaDB dan re-ranking hibrida leksikal).
- **Berkas Output:** Laporan terstruktur di `output/retrieval_benchmark.json` dan visualisasi grafik ambang batas di `output/retrieval_similarity.png`.

---

## Prasyarat dan Cara Menjalankan

### Prasyarat Sistem
- Windows, Linux, atau macOS.
- Python 3.11 atau lebih baru.
- Astral `uv` sebagai manajer dependensi Python.

### 1. Instalasi Lingkungan
```powershell
git clone https://github.com/boyharendy/Milestone-Certan-Projek.git
cd Milestone-Certan-Projek
uv sync
```

### 2. Menjalankan Seluruh Pengujian Otomatis (70 Unit Tests)
```powershell
uv run pytest
```
*Hasil: 70 passed (100% Lulus mencakup Milestone 1, 2, dan 3).*

Untuk pengujian khusus modul Milestone 3:
```powershell
uv run pytest tests/test_pdf_parser.py tests/test_chunker.py tests/test_vector_store.py tests/test_retrieval.py tests/test_benchmark_retrieval.py
```

### 3. Menjalankan Benchmark Temu Balik Semantik (Milestone 3)
```powershell
uv run python -m src.benchmark_retrieval
```

### 4. Menguji Kueri Semantik Kustom (Python CLI)
```powershell
uv run python -c "from src.knowledge_base.retriever import SemanticRetriever; r = SemanticRetriever(); res = r.retrieve('bagaimana kriteria tempat evakuasi sementara tsunami?'); print('Status:', res.status.value); print('Skor:', res.top_score); print('Dokumen Sumber:', res.items[0].source_file); print('Cuplikan:\n', res.items[0].text[:180])"
```

### 5. Membangun Ulang Basis Data Vektor ChromaDB (Opsional)
```powershell
uv run python -m src.knowledge_base.indexer --reset
```

### 6. Menjalankan Modul Milestone 1 & 2 (Pencarian Rute & Peta)
- **Pencarian Rute Graf (UCS vs A\*):**
  ```powershell
  uv run python -m src.search
  ```
- **Benchmark Algoritma Genetika (GA Alokasi Armada):**
  ```powershell
  uv run python -m src.benchmark
  ```
- **Visualisasi Peta Riil Jalan Banda Aceh:**
  ```powershell
  uv run python -m src.visualize_map
  ```

---

## Lisensi

Proyek ini didistribusikan berdasarkan lisensi MIT. Lihat berkas [LICENSE](LICENSE) untuk informasi lebih lanjut.
