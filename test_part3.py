"""
Part 3: AI text-embedding similarity (sentence-transformers + FAISS).

Unlike Part 1 (which matches on exact words via TF-IDF), this matches on
MEANING. The first run downloads a small AI model (~90MB) once, then builds
a fast search index over all 30,000 products.
"""
import pandas as pd
from app.vector_search import VectorSearch

DATA = ('data/marketing_sample_for_amazon_com-amazon_fashion_products'
        '__20200201_20200430__30k_data.ldjson')

QUERY_ID = '26d41bdc1495de290bc8e6062d927729'
N = 5

print("Loading data...")
df = pd.read_json(DATA, lines=True)
df['uniq_id'] = df['uniq_id'].astype(str)

print("Building AI embeddings + index (first run downloads a model, be patient)...")
vs = VectorSearch(df, id_col='uniq_id')

print("Searching...\n")
similar = vs.find_similar(QUERY_ID, N)

def show(pid, label):
    row = df[df['uniq_id'] == pid]
    if row.empty:
        print(f"{label}: (not found)"); return
    r = row.iloc[0]
    print(f"{label}")
    print(f"    {str(r.get('product_name',''))[:70]}")
    print(f"    brand: {r.get('brand','N/A')} | price: {r.get('sales_price','N/A')}")
    print()

show(QUERY_ID, "QUERY:")
print("-" * 60)
for i, pid in enumerate(similar, 1):
    show(pid, f"AI MATCH {i}")
