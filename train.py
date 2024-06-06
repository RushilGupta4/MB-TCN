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



outcome = os.getenv("OUTCOME")
is_adam_optimizer = os.getenv("OPTIM") == "adam"

device_name = "cuda" if torch.cuda.is_available() else "cpu"
# device_name = "mps"
device = torch.device(device_name)

print("Python Version:", sys.version)
print("Pytorch Version:", torch.__version__)
print("Numpy Version:", np.__version__)
print("Pandas Version:", pd.__version__)
print("Sklearn Version:", sklearn.__version__)

torch.cuda.empty_cache()


if device_name == "cuda":
    print("===========================================")
    print("Num GPUs Available: ", torch.cuda.device_count())
    print("===========================================")
    print("Device Information:", torch.cuda.get_device_name(0))


train_df, val_df, test_df, n_outputs = load_data(column=outcome)


train_dataset = PatientDataset(train_df)
val_dataset = PatientDataset(val_df)


num_inputs = 256
num_inputs = 1097
num_channels = [16, 16]
# n_outputs = 3
n_branches = 10
kernel_size = 10
dropout = 0.4
batch_size = 128
batch_size = 32


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True
)

model = MBTCN(
    num_inputs=num_inputs,
    num_channels=num_channels,
    n_outputs=n_outputs,
    n_branches=n_branches,
    kernel_size=kernel_size,
    dropout=dropout,
).to(device)

criterion = nn.CrossEntropyLoss()
if is_adam_optimizer:
    optimizer = optim.Adam(model.parameters())
else:
    optimizer = optim.SGD(model.parameters(), momentum=0.9)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True
)

# Training the model
num_epochs = 100
train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    MB_NUM=n_branches,
    batch_size=batch_size,
    device=device,
    outcome=outcome
)
