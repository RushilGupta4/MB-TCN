import torch
import os
import pprint

outcome = os.getenv("OUTCOME")

res = torch.load(f"mbtcn_gru_ls_{outcome}/model_800.pt")
pprint.pprint(res["losses"])