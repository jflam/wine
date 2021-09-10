#%%
import constants as C
import nmslib
import pandas as pd
import sqlite3 as sql
import time

from sentence_transformers import SentenceTransformer

# TODO: see if space should be cosinesimil_sparse
print(f"LOADING NMS index from {C.NMS_INDEX1}...")
index = nmslib.init(method="hnsw", space="cosinesimil")
index.loadIndex(C.NMS_INDEX1)

print(f"LOADING sentence transformer {C.SENTENCE_TRANSFORMER_MODEL_NAME}...")
model = SentenceTransformer(C.SENTENCE_TRANSFORMER_MODEL_NAME)

print(f"LOADING dataset from {C.SQLITE_DATASET} sqlite file...")
with sql.connect(C.SQLITE_DATASET) as c:
    df = pd.read_sql("select * from wine", c)
#%%
query = "I would like a fruity pinot noir with cherry flavors"

start = time.process_time()
query_embeddings = model.encode(query, convert_to_tensor=True).cpu()
ids, distances = index.knnQuery(query_embeddings, k=20)
end = time.process_time()
print(f"SEARCHED {df.shape[0]} reviews of {df.title.nunique()} wines from {df.winery.nunique()} wineries in {end-start:.2f} seconds\n")

matches = []
for i, j in zip(ids, distances):
    print((f"NAME: {df.winery.values[i]} {df.title.values[i]} "
           f"({df.country.values[i]})\n"
           f"REVIEW: {df.description.values[i]}\n"
           f"RANK: {df.points.values[i]} "
           f"DISTANCE: {j:.2f}"))
