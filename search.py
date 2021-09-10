#%%
import nmslib
import pandas as pd
import sqlite3 as sql

from sentence_transformers import SentenceTransformer

# TODO: see if space should be cosinesimil_sparse
print(f"LOADING index from ./data/index.bin...")
index = nmslib.init(method="hnsw", space="cosinesimil")
index.loadIndex("./data/index.bin")

print(f"LOADING distilBERT model...")
model = SentenceTransformer("distilbert-base-uncased")

print(f"LOADING dataset from ./data/wine.dat sqlite file...")
with sql.connect("./data/wine.db") as c:
    df = pd.read_sql("select * from wine", c)

print(df.head())
#%%
query = "I would like an inexpensive wine to serve at parties"

query_embeddings = model.encode(query, convert_to_tensor=True).cpu()
ids, distances = index.knnQuery(query_embeddings, k=20)
matches = []

for i, j in zip(ids, distances):
    print((f"NAME: {df.winery.values[i]} {df.title.values[i]} ({df.country.values[i]})\n"
           f"REVIEW: {df.description.values[i]}\n"
           f"RANK: {df.points.values[i]} "
           f"DISTANCE: {j:.2f}"))