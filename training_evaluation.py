import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.nn.utils import weight_norm
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    num_epochs,
    MB_NUM,
    batch_size,
):
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        train_size = len(train_loader)
        branch_train_size = train_size // MB_NUM + 1

        for i, (data) in enumerate(
            tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        ):
            active_branch = i // branch_train_size
            patient_id, masks, inputs, labels = (
                data["ID"],
                data["Mask"],
                data["Value"],
                data["Label"],
            )

            optimizer.zero_grad()
            outputs = model(inputs, masks, active_branch=active_branch)
            final_outputs = torch.mean(outputs, dim=0)

            loss = criterion(final_outputs.float(), labels.float())

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        val_loss = evaluate_model(val_loader, model, criterion, MB_NUM, device)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}\n"
        )


def evaluate_model(data_loader, model, criterion, MB_NUM, device):
    model.eval()
    total_loss = 0.0

    branch_eval_size = len(data_loader) // MB_NUM + 1

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc="Evaluating")):
            active_branch = i // branch_eval_size
            patient_id, masks, inputs, labels = (
                data["ID"],
                data["Mask"],
                data["Value"],
                data["Label"],
            )
            inputs = torch.nan_to_num(inputs).to(device)
            labels = labels.to(device)

            outputs = model(inputs, masks, active_branch=active_branch)
            final_outputs = torch.mean(outputs, dim=0)

            loss = criterion(final_outputs.float(), labels.float())
            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(data_loader.dataset)

    # print(f"Loss: {avg_loss:.4f}")

    return avg_loss
