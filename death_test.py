import pandas as pd
import numpy as np

df = pd.read_parquet("./sepsis_labelled.parquet")

# Since 200 > 30 => no death
df["death_time"] = pd.to_timedelta(df["death_time"]).fillna(pd.Timedelta(days=200))

df["death_time"] = pd.to_timedelta(np.where(
    df["death_time"] == pd.Timedelta(days=0),
    pd.Timedelta(days=200),
    df["death_time"]
))

df["death_time"] = pd.to_timedelta(np.where(
    df["death_time"] > pd.Timedelta(days=30),
    pd.Timedelta(days=200),
    df["death_time"]
))

df["death_time"] = df["death_time"].dt.total_seconds() / 3600

print(df.drop_duplicates("ID")["death_time"].value_counts())