"""
Part 1: Product similarity search.

Design summary
--------------
A product is described by a mix of attribute types, and each type needs a
different notion of "distance". We therefore build ONE feature matrix by
concatenating three blocks, each separately normalised so no single block
dominates:

  1. Numeric block  : price, sales_price, weight, rating
                      -> median-imputed, then StandardScaler (z-scores).
                      Without scaling, price (tens of dollars) would swamp
                      rating (0-5). Standardising puts every column on a
                      comparable scale.

  2. Categorical    : brand, color
                      -> one-hot encoded. Two products with the same brand
                      get a shared "1" in that column, contributing to
                      similarity. High-cardinality is controlled with
                      min_frequency so we don't explode the matrix.

  3. Text block     : product_name (+ description if present)
                      -> TF-IDF. Captures lexical overlap ("running shoe"
                      vs "running sneaker"). Part 3 upgrades this to
                      transformer embeddings for semantic similarity.

Similarity measure: cosine similarity on the combined sparse/dense matrix.
Cosine is scale-invariant in magnitude and works well with the sparse,
high-dimensional TF-IDF + one-hot space. (Euclidean on z-scored numerics
would also be fine; cosine is chosen so the same measure works across all
blocks consistently.)

We precompute the full feature matrix ONCE at load time. Each query is then
a single matrix-vector cosine op, which is cheap. Part 3 replaces the
"compare against everything" step with an ANN index for sub-linear lookup.

Trade-offs
----------
- Precomputing the matrix costs memory but makes every query fast and keeps
  the code simple. For 30k rows this is trivial.
- One-hot on brand/color is interpretable and fast but ignores that some
  colors are "closer" (navy vs blue). Acceptable for v1.
- Cosine treats all blocks with equal implicit weight after normalisation.
  We expose a weighting hook so blocks can be reweighted later.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Column resolution: the raw dataset has messy / inconsistent names, so we
# map "logical" fields to whatever actually exists in the dataframe.
# ---------------------------------------------------------------------------
#
# NOTE on this specific dataset (Amazon Fashion 2020, 30k rows):
#   - 'uniq_id', 'product_name', 'weight', 'rating' are 100% populated.
#   - 'brand' (~73%) and 'sales_price' (~90%) are usable; gaps are imputed.
#   - There is NO separate list 'price' column, only 'sales_price', so price
#     similarity is based on sales_price.
#   - 'colour' is only ~20% populated, so we deliberately leave it OUT (the
#     candidate list below omits it) to avoid adding mostly-missing noise.
#   - Images live in 'image_urls__small'/'medium'/'large' (used only by the
#     optional image-similarity extension, not the core function).
#
COLUMN_CANDIDATES = {
    "id":          ["uniq_id", "unique_id", "id"],
    "name":        ["product_name", "name", "title"],
    "brand":       ["brand", "brand_name", "manufacturer"],
    "color":       [],  # intentionally skipped: 'colour' is ~80% empty here
    "price":       ["sales_price", "price", "selling_price", "list_price", "mrp"],
    "sales_price": ["sales_price", "sale_price", "selling_price", "discounted_price"],
    "weight":      ["weight", "item_weight", "shipping_weight"],
    "rating":      ["rating", "product_rating", "average_rating", "stars"],
    "image":       ["image_urls__small", "medium", "large", "image", "image_url"],
    "description": ["description", "product_description", "about_product"],
}


def _resolve(df: pd.DataFrame, logical: str) -> Optional[str]:
    for cand in COLUMN_CANDIDATES[logical]:
        if cand in df.columns:
            return cand
    return None


_PRICE_RE = re.compile(r"[-+]?\d*\.?\d+")


def _to_number(series: pd.Series) -> pd.Series:
    """Parse messy strings like '$12.99', '1,299', '3.5 out of 5 stars'."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    def parse(v):
        if pd.isna(v):
            return np.nan
        s = str(v).replace(",", "")
        m = _PRICE_RE.search(s)
        return float(m.group()) if m else np.nan

    return series.map(parse)


class ProductSimilarity:
    """Builds the feature matrix once, answers similarity queries cheaply."""

    def __init__(
        self,
        df: pd.DataFrame,
        text_weight: float = 1.0,
        numeric_weight: float = 1.0,
        categorical_weight: float = 1.0,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.cols = {k: _resolve(self.df, k) for k in COLUMN_CANDIDATES}

        id_col = self.cols["id"]
        if id_col is None:
            raise ValueError("No id column (uniq_id) found in dataframe.")
        self.id_col = id_col

        # Map product_id -> row index for O(1) lookup.
        self.id_to_idx = {pid: i for i, pid in enumerate(self.df[id_col].astype(str))}

        self._build_matrix(text_weight, numeric_weight, categorical_weight)

    # ------------------------------------------------------------------ build
    def _build_matrix(self, w_text, w_num, w_cat):
        blocks = []

        # --- numeric ---
        num_logical = ["price", "sales_price", "weight", "rating"]
        num_cols = [self.cols[c] for c in num_logical if self.cols[c]]
        if num_cols:
            num_df = pd.DataFrame(
                {c: _to_number(self.df[c]) for c in num_cols}
            )
            num_imputed = SimpleImputer(strategy="median").fit_transform(num_df)
            num_scaled = StandardScaler().fit_transform(num_imputed)
            blocks.append(sparse.csr_matrix(num_scaled) * w_num)

        # --- categorical ---
        cat_logical = ["brand", "color"]
        cat_cols = [self.cols[c] for c in cat_logical if self.cols[c]]
        if cat_cols:
            cat_df = self.df[cat_cols].astype(str).fillna("unknown")
            ohe = OneHotEncoder(
                handle_unknown="ignore", min_frequency=5, sparse_output=True
            )
            cat_mat = ohe.fit_transform(cat_df)
            # L2-normalise so the block has unit-ish scale before weighting.
            blocks.append(_l2_normalize(cat_mat) * w_cat)

        # --- text ---
        text_parts = []
        for c in ["name", "description"]:
            if self.cols[c]:
                text_parts.append(self.df[self.cols[c]].astype(str).fillna(""))
        if text_parts:
            text = text_parts[0]
            for extra in text_parts[1:]:
                text = text + " " + extra
            tfidf = TfidfVectorizer(
                max_features=20000, stop_words="english", ngram_range=(1, 2)
            )
            text_mat = tfidf.fit_transform(text)  # already L2-normalised
            blocks.append(text_mat * w_text)

        if not blocks:
            raise ValueError("No usable feature columns found.")

        self.matrix = sparse.hstack(blocks).tocsr()

    # ------------------------------------------------------------------ query
    def find_similar(self, product_id: str, num_similar: int) -> List[str]:
        product_id = str(product_id)
        if product_id not in self.id_to_idx:
            raise KeyError(f"product_id {product_id!r} not found")

        idx = self.id_to_idx[product_id]
        query_vec = self.matrix[idx]

        # cosine against everything in one shot
        sims = cosine_similarity(query_vec, self.matrix).ravel()
        sims[idx] = -np.inf  # exclude the product itself

        # Tie-break: by sales_price (then rating). We sort by (-sim, price).
        order = self._top_k_with_tiebreak(sims, num_similar)
        return self.df.iloc[order][self.id_col].astype(str).tolist()

    def _top_k_with_tiebreak(self, sims: np.ndarray, k: int) -> np.ndarray:
        # Get a generous candidate pool, then apply deterministic tie-break.
        pool = min(len(sims), max(k * 5, k + 1))
        cand = np.argpartition(-sims, pool - 1)[:pool]

        tie_col = self.cols["sales_price"] or self.cols["rating"]
        if tie_col:
            tie_vals = _to_number(self.df[tie_col]).to_numpy()
            tie = np.nan_to_num(tie_vals[cand], nan=0.0)
        else:
            tie = np.zeros(len(cand))

        # sort primarily by similarity desc, then tie-break asc on price
        ordered = sorted(
            range(len(cand)),
            key=lambda i: (-sims[cand[i]], tie[i]),
        )
        return cand[ordered[:k]]


# ---------------------------------------------------------------------------
# Module-level singleton so the FastAPI app builds the matrix only once.
# ---------------------------------------------------------------------------
_ENGINE: Optional[ProductSimilarity] = None
DATA_PATH = (
    "data/marketing_sample_for_amazon_com-amazon_fashion_products"
    "__20200201_20200430__30k_data.ldjson"
)


def _l2_normalize(mat):
    from sklearn.preprocessing import normalize
    return normalize(mat, norm="l2", axis=1)


@lru_cache(maxsize=1)
def get_engine() -> ProductSimilarity:
    df = pd.read_json(DATA_PATH, lines=True)
    return ProductSimilarity(df)


def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    """Public API required by the exercise."""
    return get_engine().find_similar(product_id, num_similar)


def calculate_similarity():  # kept to match the requested stub
    raise NotImplementedError("Use ProductSimilarity.find_similar instead.")
