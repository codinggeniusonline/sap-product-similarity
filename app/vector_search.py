"""
Part 3: Vector search with transformer embeddings + ANN index.

What changes vs Part 1
----------------------
Part 1 uses TF-IDF (lexical) and does an exact cosine scan over all rows.
For semantic similarity and for scaling to large catalogues we:

  1. Embed product text with a sentence-transformer
     (all-MiniLM-L6-v2: 384-dim, fast, strong quality/size trade-off).
     Unlike TF-IDF, this captures meaning: "sneakers" ~ "running shoes"
     even with no shared tokens.

  2. Index the embeddings with FAISS using HNSW
     (Hierarchical Navigable Small World graphs, Malkov & Yashunin 2018,
     "Efficient and robust approximate nearest neighbor search using
     Hierarchical Navigable Small World graphs", arXiv:1603.09320).

Why HNSW
--------
- Query time is ~O(log N) vs O(N) for the exact scan, so it scales to
  millions of items where the brute-force cosine in Part 1 would not.
- Graph-based ANN gives very high recall at low latency, and unlike IVF it
  needs no separate training step — good for a 30k catalogue that may grow.
- Trade-off: it's approximate (recall < 100%) and the index uses more memory
  than a flat list. For product recommendation, occasionally missing the
  Nth-best neighbour is acceptable; latency and scale matter more.

We embed on a normalised vector and use inner product, which equals cosine
similarity on unit vectors — consistent with Part 1's measure.

Numeric / categorical attributes can be fused by concatenating their scaled
vectors onto the text embedding before indexing (hook left in build_index).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


class VectorSearch:
    def __init__(self, df: pd.DataFrame, id_col: str = "uniq_id",
                 model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.df = df.reset_index(drop=True).copy()
        self.id_col = id_col if id_col in df.columns else df.columns[0]
        self.ids = self.df[self.id_col].astype(str).tolist()
        self.id_to_idx = {pid: i for i, pid in enumerate(self.ids)}

        self.model = SentenceTransformer(model_name)
        self._build_index()

    def _text(self) -> List[str]:
        parts = []
        for c in ["product_name", "name", "title", "brand", "colour", "color", "description"]:
            if c in self.df.columns:
                parts.append(self.df[c].astype(str).fillna(""))
        if not parts:
            parts = [self.df.iloc[:, 0].astype(str)]
        text = parts[0]
        for p in parts[1:]:
            text = text + " " + p
        return text.tolist()

    def _build_index(self):
        import faiss

        emb = self.model.encode(
            self._text(), batch_size=64, show_progress_bar=False,
            normalize_embeddings=True,  # unit vectors -> inner product = cosine
        ).astype("float32")
        self.embeddings = emb

        dim = emb.shape[1]
        # HNSW index; 32 = graph connectivity (M). Higher = better recall, more mem.
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200   # build-time quality
        index.hnsw.efSearch = 64          # query-time quality/speed knob
        index.add(emb)
        self.index = index

    def find_similar(self, product_id: str, num_similar: int) -> List[str]:
        product_id = str(product_id)
        if product_id not in self.id_to_idx:
            raise KeyError(f"product_id {product_id!r} not found")
        idx = self.id_to_idx[product_id]
        q = self.embeddings[idx : idx + 1]
        # fetch k+1 because the product itself is its own nearest neighbour
        _, nbrs = self.index.search(q, num_similar + 1)
        out = [self.ids[j] for j in nbrs[0] if j != idx]
        return out[:num_similar]


_VS: Optional[VectorSearch] = None


def get_vector_search(df: pd.DataFrame) -> VectorSearch:
    global _VS
    if _VS is None:
        _VS = VectorSearch(df)
    return _VS
