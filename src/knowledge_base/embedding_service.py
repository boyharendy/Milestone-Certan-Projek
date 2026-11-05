"""Modul layanan embedding dense vector menggunakan Sentence-Transformers.

Menyediakan:
- Base class abstrak untuk embedding.
- SentenceTransformerEmbeddingService untuk inferensi model multilingual.
- DummyEmbeddingService untuk pengujian unit deterministik & cepat.
"""

from abc import ABC, abstractmethod
import hashlib
import math
from typing import List, Optional


class BaseEmbeddingService(ABC):
    """Antarmuka dasar untuk layanan dense vector embedding."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Mengubah daftar teks menjadi daftar vektor berdimensi tetap."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Mengubah satu kueri menjadi vektor embedding."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensi vektor keluaran."""
        pass


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Layanan embedding menggunakan pustaka sentence-transformers.

    Default model: 'paraphrase-multilingual-MiniLM-L12-v2'
    - Mendukung 50+ bahasa (termasuk Bahasa Indonesia formal).
    - Dimensi output: 384 dimensi.
    - Cepat, akurat, dan ringan untuk inferensi CPU/GPU lokal.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy loading model untuk efisiensi memori."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Menghasilkan embedding untuk kumpulan teks dalam batch terukur."""
        if not texts:
            return []

        self._load_model()
        # normalize_embeddings=True agar dot product setara dengan cosine similarity
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Menghasilkan embedding tunggal untuk teks kueri."""
        results = self.embed_texts([text])
        return results[0]

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension or 384


class DummyEmbeddingService(BaseEmbeddingService):
    """Layanan embedding tiruan untuk pengujian unit yang cepat dan deterministik."""

    def __init__(self, dimension: int = 4):
        self._dim = dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        # Bangkitkan vektor deterministik berdasarkan hash teks
        raw = []
        for i in range(self._dim):
            h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).hexdigest()
            val = (int(h[:8], 16) / 0xFFFFFFFF) - 0.5
            raw.append(val)

        # L2 Normalization
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)

    @property
    def dimension(self) -> int:
        return self._dim

