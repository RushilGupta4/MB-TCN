import os
import torch
from tqdm import tqdm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f"\n\nDEVICE: {device}\n\n")

def train_model(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    num_epochs,
    MB_NUM,
    batch_size,
    device,
    outcome
):
    best_val_loss = float("inf")
    best_model_state = None
    optimizer_name = os.getenv("OPTIM")

    epoch_to_save = 5

    losses = []

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
            outputs = model(inputs.to(device), masks.to(device), active_branch=active_branch)
            final_outputs = torch.mean(outputs, dim=0)

            loss = criterion(final_outputs.float(), labels.to(device).float())

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        val_loss = evaluate_model(val_loader, model, criterion, MB_NUM, device)

        losses.append({
            "Train": avg_train_loss,
            "Validation": val_loss
        })

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}\n"
        )

        if (epoch + 1) % epoch_to_save == 0:
            folder = f"mbtcn_{outcome}_{optimizer_name}"
            os.makedirs(folder, exist_ok=True)
            model_path = os.path.join(folder, f"model_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "losses": losses,
                },
                model_path,
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
            masks = masks.to(device)

            outputs = model(inputs, masks, active_branch=active_branch)
            final_outputs = torch.mean(outputs, dim=0)

            loss = criterion(final_outputs.float(), labels.float())
            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(data_loader.dataset)

    # print(f"Loss: {avg_loss:.4f}")

    return avg_loss
