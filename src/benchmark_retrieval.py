"""Skrip evaluasi dan benchmark temu balik semantik korpus kebencanaan (Milestone 3).

Tanggung jawab utama (Role 3: Dianita Lorensia Br Ginting):
1. Menguji matriks kueri bisnis (Spesifik/Regulasi, Parafrase/Bahasa Alami, Ambigu, dan Out-of-Domain).
2. Mengukur metrik kinerja: Cosine Similarity, Latensi (ms), dan Presisi Penolakan OOD.
3. Menyimpan hasil terstruktur ke `output/retrieval_benchmark.json`.
4. Menghasilkan visualisasi analisis batas ambang batas (threshold cutoff) ke `output/retrieval_similarity.png`.

Dijalankan secara langsung dengan:
    uv run python -m src.benchmark_retrieval
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List
import matplotlib.pyplot as plt

from src.knowledge_base.retriever import SemanticRetriever, QueryStatus
from src.knowledge_base.vector_store import VectorStore


# Matriks kueri uji bisnis kebencanaan terstruktur
TEST_QUERY_SUITE = [
    # 1. Kueri Spesifik / Regulasi & Teknis In-Domain
    {
        "id": "Q1",
        "category": "Spesifik",
        "query": "prosedur aktivasi sirine peringatan dini tsunami",
        "expected_status": "accepted",
        "description": "Istilah teknis SOP Pusdalops PB / Sistem peringatan dini",
    },
    {
        "id": "Q2",
        "category": "Spesifik",
        "query": "kriteria struktur dan kapasitas Tempat Evakuasi Sementara TES tsunami",
        "expected_status": "accepted",
        "description": "Standar teknis bangunan vertikal TES",
    },
    {
        "id": "Q3",
        "category": "Spesifik",
        "query": "tugas dan wewenang BPBA dalam penyelenggaraan penanggulangan bencana",
        "expected_status": "accepted",
        "description": "Pasal regulasi Qanun / Pergub Aceh",
    },
    {
        "id": "Q4",
        "category": "Spesifik",
        "query": "instruksi presiden penanganan darurat dan rekonstruksi pasca tsunami NAD",
        "expected_status": "accepted",
        "description": "Inpres No. 1 Tahun 2005 tentang rehab-rekon pasca tsunami",
    },
    # 2. Kueri Parafrase / Bahasa Alami Non-Literal In-Domain
    {
        "id": "Q5",
        "category": "Parafrase",
        "query": "tempat aman berlindung saat ombak laut pasang besar di Banda Aceh",
        "expected_status": "accepted",
        "description": "Bahasa awam tanpa kata eksplisit 'tsunami'",
    },
    {
        "id": "Q6",
        "category": "Parafrase",
        "query": "apa yang harus dilakukan warga pesisir saat merasakan gempa bumi kuat",
        "expected_status": "accepted",
        "description": "Prosedur evakuasi mandiri berbasis skenario masyarakat",
    },
    # 3. Kueri Ambigu / Parsial In-Domain
    {
        "id": "Q7",
        "category": "Ambigu",
        "query": "kapasitas evakuasi tsunami",
        "expected_status": "accepted",
        "description": "Kueri singkat multi-interpretasi",
    },
    {
        "id": "Q8",
        "category": "Ambigu",
        "query": "jalur logistik darurat",
        "expected_status": "accepted",
        "description": "Manajemen pendistribusian logistik darurat",
    },
    # 4. Kueri Out-of-Domain (OOD / Distractor)
    {
        "id": "Q9",
        "category": "Out-of-Domain",
        "query": "resep membuat mie aceh tradisional bumbu rempah kepiting",
        "expected_status": "rejected_out_of_domain",
        "description": "Topik kuliner di luar domain kebencanaan",
    },
    {
        "id": "Q10",
        "category": "Out-of-Domain",
        "query": "harga tiket bioskop Hermes Palace XXI Banda Aceh",
        "expected_status": "rejected_out_of_domain",
        "description": "Topik hiburan komersial non-kebencanaan",
    },
    {
        "id": "Q11",
        "category": "Out-of-Domain",
        "query": "cara memperbaiki laptop windows mengalami blue screen of death",
        "expected_status": "rejected_out_of_domain",
        "description": "Topik perangkat keras / IT umum",
    },
    {
        "id": "Q12",
        "category": "Out-of-Domain",
        "query": "analisis pergerakan saham perbankan di bursa efek indonesia",
        "expected_status": "rejected_out_of_domain",
        "description": "Topik finansial dan pasar modal",
    },
]


def run_retrieval_benchmark(
    threshold: float = 0.50,
    top_k: int = 3,
    output_dir: str = "output",
) -> Dict[str, Any]:
    """Menjalankan evaluasi temu balik semantik dan menghasilkan output terstruktur."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("BENCHMARK TEMU BALIK SEMANTIK & EVALUASI OOD (MILESTONE 3)")
    print("=" * 75)
    print(f"Ambang Batas Similarity (Threshold): {threshold}")
    print(f"Top-K Dokumen                       : {top_k}")
    print(f"Total Kueri Uji                     : {len(TEST_QUERY_SUITE)}")
    print("=" * 75)

    vector_store = VectorStore()
    retriever = SemanticRetriever(vector_store=vector_store, threshold=threshold)

    results: List[Dict[str, Any]] = []
    in_domain_scores: List[float] = []
    ood_scores: List[float] = []
    latencies: List[float] = []

    correct_predictions = 0

    for item in TEST_QUERY_SUITE:
        qid = item["id"]
        category = item["category"]
        query_text = item["query"]
        expected = item["expected_status"]

        start_time = time.perf_counter()
        retrieval_res = retriever.retrieve(query=query_text, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        latencies.append(elapsed_ms)
        top_score = retrieval_res.top_score
        actual = retrieval_res.status.value

        is_correct = actual == expected
        if is_correct:
            correct_predictions += 1

        if category == "Out-of-Domain":
            ood_scores.append(top_score)
        else:
            in_domain_scores.append(top_score)

        top_source = (
            f"{retrieval_res.items[0].source_file} (p.{retrieval_res.items[0].page_number})"
            if retrieval_res.items
            else "None"
        )

        status_flag = "[PASS]" if is_correct else "[FAIL]"
        print(
            f"{status_flag} {qid} [{category:<13}] Sim: {top_score:.4f} | Latency: {elapsed_ms:6.2f}ms | Status: {actual}"
        )
        print(f"       Kueri : \"{query_text}\"")
        if retrieval_res.items:
            print(f"       TopDoc: {top_source}")
        elif retrieval_res.rejection_reason:
            print(f"       Alasan: {retrieval_res.rejection_reason}")
        print("-" * 75)

        results.append(
            {
                "id": qid,
                "category": category,
                "query": query_text,
                "description": item["description"],
                "expected_status": expected,
                "actual_status": actual,
                "is_correct": is_correct,
                "top_similarity_score": round(top_score, 4),
                "latency_ms": round(elapsed_ms, 2),
                "top_source": top_source,
                "rejection_reason": retrieval_res.rejection_reason,
            }
        )

    # Hitung metrik agregat
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_in_domain = sum(in_domain_scores) / len(in_domain_scores) if in_domain_scores else 0.0
    avg_ood = sum(ood_scores) / len(ood_scores) if ood_scores else 0.0
    accuracy = (correct_predictions / len(TEST_QUERY_SUITE)) * 100.0

    in_domain_total = len(in_domain_scores)
    in_domain_correct = sum(
        1 for r in results if r["category"] != "Out-of-Domain" and r["is_correct"]
    )
    in_domain_recall = (in_domain_correct / in_domain_total * 100.0) if in_domain_total else 0.0

    ood_total = len(ood_scores)
    ood_correct = sum(
        1 for r in results if r["category"] == "Out-of-Domain" and r["is_correct"]
    )
    ood_precision = (ood_correct / ood_total * 100.0) if ood_total else 0.0

    summary = {
        "total_queries": len(TEST_QUERY_SUITE),
        "overall_accuracy_percent": round(accuracy, 2),
        "in_domain_recall_percent": round(in_domain_recall, 2),
        "ood_rejection_precision_percent": round(ood_precision, 2),
        "similarity_threshold": threshold,
        "avg_in_domain_similarity": round(avg_in_domain, 4),
        "avg_ood_similarity": round(avg_ood, 4),
        "score_margin": round(avg_in_domain - avg_ood, 4),
        "avg_latency_ms": round(avg_latency, 2),
    }

    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "vector_store": "ChromaDB (PersistentClient)",
            "collection": "disaster_knowledge_base",
            "total_indexed_chunks": vector_store.count(),
        },
        "summary": summary,
        "query_details": results,
    }

    # 1. Simpan laporan JSON
    json_path = out_path / "retrieval_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Laporan JSON tersimpan di: {json_path}")

    # 2. Buat visualisasi grafik analisis ambang batas
    fig_path = out_path / "retrieval_similarity.png"
    _generate_similarity_plot(results, threshold, fig_path)
    print(f"[OK] Grafik visualisasi tersimpan di: {fig_path}")

    print("\n" + "=" * 75)
    print("RINGKASAN METRIK EVALUASI TEMU BALIK")
    print("=" * 75)
    print(f"Total Dokumen di Database   : {vector_store.count()} chunks")
    print(f"Akurasi Keseluruhan         : {accuracy:.2f}% ({correct_predictions}/{len(TEST_QUERY_SUITE)})")
    print(f"In-Domain Recall            : {in_domain_recall:.2f}% ({in_domain_correct}/{in_domain_total})")
    print(f"OOD Rejection Precision     : {ood_precision:.2f}% ({ood_correct}/{ood_total})")
    print(f"Rata-rata Kemiripan In-Domain: {avg_in_domain:.4f}")
    print(f"Rata-rata Kemiripan OOD      : {avg_ood:.4f}")
    print(f"Margin Pemisahan (Gap)       : {summary['score_margin']:.4f}")
    print(f"Rata-rata Latensi Inferensi : {avg_latency:.2f} ms")
    print("=" * 75)

    return report


def _generate_similarity_plot(
    results: List[Dict[str, Any]],
    threshold: float,
    output_image_path: Path,
) -> None:
    """Menghasilkan plot diagram batang skor Cosine Similarity dengan garis ambang batas."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6.5))

    query_labels = [r["id"] for r in results]
    scores = [r["top_similarity_score"] for r in results]
    categories = [r["category"] for r in results]

    # Palet warna berdasarkan kategori kueri
    color_map = {
        "Spesifik": "#1f77b4",       # Biru
        "Parafrase": "#2ca02c",      # Hijau
        "Ambigu": "#ff7f0e",         # Oranye
        "Out-of-Domain": "#d62728",   # Merah
    }
    bar_colors = [color_map.get(cat, "#7f7f7f") for cat in categories]

    bars = ax.bar(query_labels, scores, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)

    # Tambahkan label skor di atas setiap batang
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(
            f"{score:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Garis ambang batas (Threshold Cutoff)
    ax.axhline(
        y=threshold,
        color="crimson",
        linestyle="--",
        linewidth=2.0,
        label=f"Threshold Cutoff (τ = {threshold:.2f})",
    )

    # Area arsiran penolakan OOD
    ax.axhspan(0.0, threshold, color="red", alpha=0.08, label="Zona Penolakan Kueri (OOD / Ditolak)")

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("ID Kueri Uji Bisnis", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cosine Similarity Score (Top-1)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Distribusi Skor Cosine Similarity Kueri Kebencanaan vs Out-of-Domain (Milestone 3)\n"
        "Model: paraphrase-multilingual-MiniLM-L12-v2 | Basis Data: ChromaDB Persisten",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    # Legend kustom
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="In-Domain: Regulasi / Spesifik"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="In-Domain: Parafrase / Bahasa Alami"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="In-Domain: Ambigu / Singkat"),
        Patch(facecolor="#d62728", edgecolor="black", label="Out-of-Domain (OOD Distractor)"),
        Line2D([0], [0], color="crimson", linestyle="--", linewidth=2.0, label=f"Ambang Batas (τ = {threshold:.2f})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    fig.savefig(output_image_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run_retrieval_benchmark()

