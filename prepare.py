import constants as C
import os
import pandas as pd
import sqlite3 as sql

# Only clean the data if the data/wine.db doesn't exist already
if not os.path.isfile(C.SQLITE_DATASET):
    if not os.path.isfile(C.CSV_DATASET):
        print(f"Missing {C.CSV_DATASET} file. You can download it from: "
               "https://www.kaggle.com/zynicide/wine-reviews. Extract the "
               "winmag-data-130k-v2.csv file from the archive and put it in "
              f"{C.CSV_DATASET}")
        exit(1)
    df = pd.read_csv(C.CSV_DATASET)
    print(f"CLEANING data, original shape = {df.shape}")
    df = df.drop_duplicates("description")
    df = df.dropna(subset=["price"])
    df = df.groupby("variety").filter(lambda x : len(x) > 200)
    print(f"DONE CLEANING, new shape is {df.shape}")
    print(df.head())

    # Persist as sqlite3
    print(f"SAVING data to {C.SQLITE_DATASET} sqlite database")
    with sql.connect(C.SQLITE_DATASET) as c:
        df.to_sql('wine', c)

# Creates a new index from the wine reviews dataset 
if (os.path.isfile(C.NMS_INDEX1) and 
    os.path.isfile(C.C.NMS_INDEX2)):
    print("INDEX files already exist. Nothing done.")
    exit(0)

#%% 
import nmslib
import time

from sentence_transformers import SentenceTransformer

#%%
print(f"DOWNLOADING {C.SENTENCE_TRANSFORMER_MODEL_NAME} model...")
model = SentenceTransformer(C.SENTENCE_TRANSFORMER_MODEL_NAME)
with sql.connect(C.SQLITE_DATASET) as c:
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
index.saveIndex(C.NMS_INDEX1, save_data=True)

print("DONE")