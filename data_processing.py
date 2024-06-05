import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class PatientDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # Retrieve data for a single patient indexed by `index`.
        data = self.data_list[index]
        patient_id = data["ID"]

        # Convert data into tensors and reshape for TCN input requirements.
        masks = torch.tensor(data["Mask"], dtype=torch.float32).transpose(0, 1)
        values = torch.tensor(data["Value"], dtype=torch.float32).transpose(0, 1)
        label = torch.tensor(data["Label"], dtype=torch.int32)
        # label = torch.argmax(label, dim=-1)
        times = torch.tensor(data["Times"], dtype=torch.float32)

        return {
            "ID": patient_id,
            "Mask": masks,
            "Value": values,
            "Label": label,
            "Times": times,
        }


def load_data(column: str):
    """
    download data and save as list format, each element in list is a dataframe

    Args:
    column: The label column to use for classification

    """

    df = pd.read_parquet("../ckpt5_zscore_cut.parquet")

    # Since 200 > 30 => no death
    df["death_time"] = pd.to_timedelta(df["death_time"]).fillna(pd.Timedelta(days=200))
    df["death_time"] = df["death_time"].dt.total_seconds() / 3600
    ids = df["ID"].unique()

    split1 = int(len(ids) * 0.7)
    split2 = int(len(ids) * 0.8)

    train = ids[:split1]
    val = ids[split1:split2]
    test = ids[split2:]

    train_df = df[df["ID"].isin(train)]
    val_df = df[df["ID"].isin(val)]
    test_df = df[df["ID"].isin(test)]

    data = {}

    for i, df in {"train": train_df, "val": val_df, "test": test_df}.items():
        # Binning and one-hot encoding for each DataFrame split
        if column == "los":
            bins = [0, 48, 168, 720, float("inf")]
            labels = [0, 1, 2, 3]
            binned = pd.cut(df[column], bins=bins, labels=labels, right=False)
        elif column == "death_time":
            # Adjust binning to handle 'no death' as a separate category
            bins = [0, 48, 168, 720, float("inf")]
            labels = [0, 1, 2, 3]
            binned = pd.cut(df[column], bins=bins, labels=labels, right=False)
        else:
            raise ValueError(f"Invalid column: {column}")

        df = df.drop(columns=["death_time", "los"])
        encoded = pd.get_dummies(binned, prefix="y").astype(int)
        df = pd.concat([df, encoded], axis=1)

        patient_data = []
        for id, sub_df in df.groupby("ID"):
            max_length = 49
            mask = np.pad(
                sub_df[[c for c in sub_df.columns if c.startswith("Mask_")]].values,
                ((0, max_length - len(sub_df)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            value = np.pad(
                sub_df[[c for c in sub_df.columns if c.startswith("Value_")]].values,
                ((0, max_length - len(sub_df)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            label = sub_df[["y_0", "y_1", "y_2", "y_3"]].values[0]
            time = np.pad(
                sub_df["Time"].values,
                (0, max_length - len(sub_df)),
                mode="constant",
                constant_values=0,
            )

            item = {
                "ID": id,
                "Mask": mask,
                "Value": value,
                "Label": label,
                "Times": time,
            }
            patient_data.append(item)

        data[i] = patient_data

    return data["train"], data["val"], data["test"]
