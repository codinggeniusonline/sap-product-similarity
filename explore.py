"""
Fuller look - prints the complete column list clearly so we can map
the code to the real names.
"""
import pandas as pd

DATA = ('data/marketing_sample_for_amazon_com-amazon_fashion_products'
        '__20200201_20200430__30k_data.ldjson')

df = pd.read_json(DATA, lines=True)

print("TOTAL COLUMNS:", df.shape[1])
print("TOTAL ROWS:", df.shape[0])
print()
print("=== FULL COLUMN LIST (numbered) ===")
for i, c in enumerate(df.columns.tolist()):
    # also show how many values are NOT empty, so we know which columns are usable
    non_empty = df[c].notna().sum()
    print(f"{i+1:>2}. {c:<45} ({non_empty} non-empty)")

print()
print("=== ID COLUMN + EXAMPLE ID ===")
id_col = 'uniq_id' if 'uniq_id' in df.columns else df.columns[0]
print("id column:", id_col)
print("example id:", df[id_col].iloc[0])
