"""
Shows the query product, then its 5 most similar matches side by side,
so YOU can eyeball whether the matches make sense.
"""
import pandas as pd
from app.similarity import get_engine, find_similar_products

DATA = ('data/marketing_sample_for_amazon_com-amazon_fashion_products'
        '__20200201_20200430__30k_data.ldjson')

# Pick a product to test. Change this to any uniq_id you like.
QUERY_ID = '26d41bdc1495de290bc8e6062d927729'
N = 5

df = pd.read_json(DATA, lines=True)
df['uniq_id'] = df['uniq_id'].astype(str)

def show(pid, label):
    row = df[df['uniq_id'] == pid]
    if row.empty:
        print(f"{label}: (id {pid} not found)")
        return
    r = row.iloc[0]
    name = str(r.get('product_name', ''))[:70]
    brand = r.get('brand', 'N/A')
    price = r.get('sales_price', 'N/A')
    rating = r.get('rating', 'N/A')
    print(f"{label}")
    print(f"    name  : {name}")
    print(f"    brand : {brand}  |  price: {price}  |  rating: {rating}")
    print()

print("=" * 75)
print("QUERY PRODUCT")
print("=" * 75)
show(QUERY_ID, "INPUT:")

print("=" * 75)
print(f"TOP {N} SIMILAR PRODUCTS")
print("=" * 75)
similar = find_similar_products(QUERY_ID, N)
for i, pid in enumerate(similar, 1):
    show(pid, f"MATCH {i}  (id: {pid})")
