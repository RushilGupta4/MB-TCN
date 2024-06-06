import json
from data_processing import *
from MBTCN_Module import *
from training_evaluation import *

import sys
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

outcome = os.getenv("OUTCOME")
optimizer = os.getenv("OPTIM")

ckpt_data = {
    "adam": {
        "sepsis": 35,
        "los": 20,
        "death_time": 30
    },
    "sgd": {
        "sepsis": 10,
        "los": 100,
        "death_time": 100
    }
}

device_name = "cuda" if torch.cuda.is_available() else "cpu"
# device_name = "mps"
device = torch.device(device_name)

# print("Python Version:", sys.version)
# print("Pytorch Version:", torch.__version__)
# print("Numpy Version:", np.__version__)
# print("Pandas Version:", pd.__version__)
# print("Sklearn Version:", sklearn.__version__)

torch.cuda.empty_cache()

# print(test_df)

# num_inputs = 256
num_inputs = 1097
num_channels = [16, 16]
# n_outputs = 3
n_branches = 10
kernel_size = 10
dropout = 0.4
batch_size = 128
batch_size = 5


_, _, test_df, n_outputs = load_data(column=outcome, test_only=True)

test_dataset = PatientDataset(test_df)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

print(f"\n\n\n============= OUTCOME : {outcome} =============")


model = MBTCN(
    num_inputs=num_inputs,
    num_channels=num_channels,
    n_outputs=n_outputs,
    n_branches=n_branches,
    kernel_size=kernel_size,
    dropout=dropout,
)

model.load_state_dict(torch.load(f"mbtcn_{outcome}_{optimizer}/model_{ckpt_data[optimizer][outcome]}.pt", map_location="cpu")['model'])
# model = torch.load(f"mbtcn_{outcome}/model_1000.pt")
model = model.to(device)



def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    branch_train_size = len(dataloader) // n_branches + 1

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            active_branch = i // branch_train_size

            patient_id, masks, inputs, labels = (
                data["ID"],
                data["Mask"],
                data["Value"],
                data["Label"],
            )

            inputs = torch.nan_to_num(inputs).to(device)
            labels = labels.to(device).float()
            masks = masks.to(device).float()

            # print(labels_indices)
            outputs = model(inputs, masks, active_branch=active_branch)
            final_outputs = torch.mean(outputs, dim=0)

            # print(patient_id, labels, labels_indices, outputs)

            # loss = criterion(final_outputs.float(), labels.float())
            loss = criterion(final_outputs, labels)
            total_loss += loss.item()

            # _, predicted = torch.max(final_outputs.data, 1)
            all_preds.extend(final_outputs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            # print(final_outputs, labels)

    # Calculate loss
    avg_loss = total_loss / len(dataloader)

    # print(len(all_targets))
    # print(len(all_preds))

    # Calculate confusion matrix and extract TP, TN, FP, FN
    # cm = confusion_matrix(all_targets, all_preds)
    # tn, fp, fn, tp = cm.ravel()

    # Calculate accuracy, precision, recall, and F1 score
    # accuracy = accuracy_score(all_targets, all_preds)
    # precision = precision_score(all_targets, all_preds)
    # recall = recall_score(all_targets, all_preds)
    # f1 = f1_score(all_targets, all_preds)

    # cm = confusion_matrix(all_targets, all_preds)
    # # print(cm)
    # # print(cm.size)
    # tp = np.diag(cm).sum()
    # fp = cm.sum(axis=0) - np.diag(cm)
    # fn = cm.sum(axis=1) - np.diag(cm)
    # tn = cm.sum() - (fp + fn + tp)

    # fp = fp.sum()
    # fn = fn.sum()
    # tn = tn.sum()  # Since each TN is counted multiple times, reduce to actual
    # # Calculate accuracy, precision, recall, and F1 score
    # accuracy = accuracy_score(all_targets, all_preds)
    # precision = precision_score(all_targets, all_preds, average='macro')
    # recall = recall_score(all_targets, all_preds, average='macro')
    # f1 = f1_score(all_targets, all_preds, average='macro')

    return {
        "all_targets": all_targets,
        "all_preds": all_preds,
        "avg_loss": avg_loss
    }

criterion = nn.CrossEntropyLoss()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Evaluate the model
evaluation_results = evaluate_model(model, test_loader, criterion, device)
os.makedirs("mbtcn_evals", exist_ok=True)
with open(f"mbtcn_evals/{os.getenv('HOUR_CAP')}_{outcome}_{optimizer}.json", "w") as f:
    json.dump(evaluation_results, f, cls=NpEncoder)
# Print results
# for key, value in evaluation_results.items():
#     print(f"{key}: {value}")

print(f"============= OUTCOME : {outcome} =============")
