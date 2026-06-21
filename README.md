# Product Similarity Search — SAP Technical Exercise

A product similarity search system built on the Amazon Fashion 2020 dataset
(~30,000 products). Given a product's `uniq_id`, it returns the most similar
products based on text, numeric, and categorical attributes.

The submission covers all three required parts:

- **Part 1** — core `find_similar_products` function (`app/similarity.py`)
- **Part 2** — FastAPI microservice + Dockerfile (`app/main.py`, `Dockerfile`)
- **Part 3** — transformer-embedding vector search with FAISS (`app/vector_search.py`)

---

## Project layout

```
.
├── app/
│   ├── similarity.py      # Part 1: feature engineering + cosine similarity
│   ├── main.py            # Part 2: FastAPI service
│   └── vector_search.py   # Part 3: sentence-transformers + FAISS HNSW
├── data/                  # the .ldjson dataset goes here (see "Getting the data")
├── explore.py             # data inspection helper
├── show_results.py        # human-readable result viewer
├── test_part3.py          # quick Part 3 check
├── requirements.txt
├── Dockerfile             # Part 2: container for k8s deployment
└── k8s.yaml               # optional Kubernetes manifest
```

---

## Getting the data

The dataset is not committed to this repository (it is ~70MB). Download it from
Kaggle and place the `.ldjson` file in a `data/` folder at the project root:

1. Download from: https://www.kaggle.com/datasets/promptcloud/amazon-fashion-products-2020
2. Unzip it. You will get a file named
   `marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson`
3. Place it at:
   `data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson`

The code reads from that path (see `DATA_PATH` in `app/similarity.py`).

---

## How to run

```bash
# 1. set up an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. inspect the data (optional but recommended)
python explore.py

# 3. Part 1 — call the function directly
python -c "from app.similarity import find_similar_products; print(find_similar_products('<uniq_id>', 5))"

# 4. Part 2 — run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
#   then open http://localhost:8000/docs

# 5. Part 3 — embedding-based search
python test_part3.py
```

Docker:

```bash
docker build -t product-similarity .
docker run -p 8000:8000 product-similarity
# open http://localhost:8000/docs
```

---

## Design decisions and reasoning

### Step 0 — data inspection first
Before writing any similarity logic I inspected the dataset (`explore.py`).
Key findings that shaped the design:

- `uniq_id`, `product_name`, `weight`, `rating` are 100% populated.
- `brand` (~73%) and `sales_price` (~90%) are usable; missing values imputed.
- There is **no separate list price** column — only `sales_price` — so price
  similarity uses `sales_price`.
- `colour` is only ~20% populated, so it is **deliberately excluded** to avoid
  injecting a mostly-missing, noisy feature.
- Several columns (`seller_name`, `no__of_reviews`, etc.) are too sparse to use.

Letting the data drive feature selection, rather than blindly using every
listed attribute, is the main design judgement here.

### Part 1 — similarity function
A product mixes three attribute types, each needing different treatment, so the
feature matrix is built from three separately-normalised blocks:

| Block | Fields | Encoding | Why |
|-------|--------|----------|-----|
| Numeric | sales_price, weight, rating | median impute → StandardScaler | Different units/ranges; scaling stops price from dominating rating |
| Categorical | brand | one-hot (rare values grouped) | Equality-based: same brand → shared signal |
| Text | product_name | TF-IDF (1–2 grams) | Captures lexical overlap between product names |

Each block is L2-normalised and the blocks are concatenated into one matrix.
Similarity is **cosine similarity**, chosen because it is scale-invariant and
works well in the sparse, high-dimensional space created by TF-IDF + one-hot.
The matrix is built **once** at load time, so each query is a single fast
matrix-vector operation. Ties are broken deterministically on `sales_price`.

### Part 2 — microservice
`GET /find_similar_products?product_id=...&num_similar=...`, built with FastAPI.
The similarity engine is warmed once at startup (FastAPI lifespan) so requests
are fast. Errors map to proper HTTP codes: **404** for an unknown product_id,
**422** for invalid `num_similar`, **500** otherwise. A `/health` endpoint
backs the Kubernetes liveness/readiness probes. The `Dockerfile` produces a
container that runs the service; `k8s.yaml` is an optional deployment manifest.

### Part 3 — vector search
`app/vector_search.py` upgrades the text comparison from lexical to **semantic**
using `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim embeddings) and
indexes the vectors with **FAISS HNSW** (Malkov & Yashunin, 2018,
*"Efficient and robust approximate nearest neighbor search using Hierarchical
Navigable Small World graphs"*, arXiv:1603.09320).

Why HNSW: it gives approximate nearest-neighbour lookup in roughly O(log N)
time, so it scales to millions of products where Part 1's exact O(N) scan would
not. It needs no training step (unlike IVF), which suits a catalogue that may
grow. Embeddings are L2-normalised so inner product equals cosine similarity —
consistent with Part 1's measure. The trade-off is that results are approximate
(recall < 100%) and the index uses more memory than a flat list, which is an
acceptable trade for the latency and scale gains in a recommendation setting.

In practice the embeddings produced visibly better matches: for a black saree
blouse fabric, the model surfaced the same product in a different colour that
the TF-IDF approach ranked lower — semantic matching over lexical matching.

---

## Known limitations / further work

- **Duplicate listings**: the dataset contains near-identical products under
  different IDs, so a query can return an exact duplicate. Near-duplicate
  suppression (e.g. dropping matches with identical names) would refine this.
- **Colour ignored**: dropped due to sparsity; with a better-populated colour
  field it would be a useful categorical feature.
- **Equal block weights**: blocks are weighted equally after normalisation. The
  code exposes weight hooks (`text_weight`, `numeric_weight`,
  `categorical_weight`) so a product owner could tune the balance.
- **Image similarity** (optional extension): image URLs are available; a
  pretrained CNN (ResNet/EfficientNet) could add a visual-similarity block,
  fused with a configurable weight and a fallback when images are missing.
