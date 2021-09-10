import os
import pandas as pd
import sqlite3 as sql

# Only clean the data if the data/wine.db doesn't exist already
if not os.path.isfile("./data/wine.db"):
    if not os.path.isfile("./data/data.csv"):
        print(("Missing data.csv file. You can download it from Kaggle: "
            "https://www.kaggle.com/zynicide/wine-reviews. Extract the "
            "winmag-data-130k-v2.csv file from the archive and put it in "
            "data/data.csv"))
        exit(1)
    df = pd.read_csv("./data/data.csv")
    print(f"CLEANING data, original shape = {df.shape}")
    df = df.drop_duplicates("description")
    df = df.dropna(subset=["price"])
    df = df.groupby("variety").filter(lambda x : len(x) > 200)
    print(f"DONE CLEANING, new shape is {df.shape}")
    print(df.head())

    # Persist as sqlite3
    print("SAVING data to wine.db sqlite database")
    with sql.connect("./data/wine.db") as c:
        df.to_sql('wine', c)