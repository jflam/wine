#%%
# This script creates a new index from the wine reviews dataset that was 
# generated by the clean.py script.

import os
if (os.path.isfile("./data/index.bin") and 
    os.path.isfile("./data/index.bin.dat")):
    print("INDEX files already exist")
    exit(0)

#%% 
import nmslib
import pandas as pd
import sqlite3 as sql
import time

from sentence_transformers import SentenceTransformer

#%%
print("RETRIEVING dataset from sqlite...")
model = SentenceTransformer("distilbert-base-uncased")
with sql.connect("./data/wine.db") as c:
    df = pd.read_sql("select * from wine", c)

#%%
# This takes about five minutes to run on an RTX 2080 with 8GB GDDR6
print(f"GENERATING embeddings from {df.shape[0]} wine descriptions")
start = time.process_time()
embeddings = model.encode(df["description"], convert_to_tensor=True)
end = time.process_time()
print(f"It took {(end-start):.2f} seconds to generate dataset embeddings")
print((f"There is one embedding per row in the dataframe. "
       f"Dataframe shape = {df.shape}, # Embeddings = {len(embeddings)}. "
       f"Each embedding is a vector of length {len(embeddings[0])}"))

# %%
# Create a new column in the dataframe to hold the embeddings
print("CREATING new index and writing it to ./data/index.bin[.dat]")
index = nmslib.init(method="hnsw", space="cosinesimil")
index.addDataPointBatch(embeddings.cpu())
index.createIndex({"post": 2}, print_progress=True)
index.saveIndex("./data/index.bin", save_data=True)