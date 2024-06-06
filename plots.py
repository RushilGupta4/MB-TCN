import matplotlib.pyplot as plt
import torch
import pandas as pd

for optim in ["adam", "sgd"]:
    for model in ["los", "sepsis", "death_time"]:
        try:
            file = f"mbtcn_{model}_{optim}/model_100.pt"
            losses = torch.load(file)['losses']
            df = pd.DataFrame(losses)

            plt.figure()  # Create a new figure for each model
            for col in df.columns:
                plt.plot(df[col], label=col)
            
            plt.legend()  # Add a legend to the plot
            plt.title(f'Losses during Training for {model}')  # Optional: add a title
            plt.xlabel('Epochs')  # Optional: label for x-axis
            plt.ylabel('Loss')  # Optional: label for y-axis
            plt.savefig(f"mbtcn_{model}_{optim}_plot.png")
            plt.clf()  # Clear the current figure after saving it
        except Exception as e:
            continue