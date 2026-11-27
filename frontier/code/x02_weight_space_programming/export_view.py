"""Write view.json: the weights the episode shows on screen (from x01's model.pt)."""
import json

import torch

s = torch.load("../x01_the_frozen_model/model.pt")
json.dump({"ln1_gain": s["blocks.0.ln1.weight"].tolist(),
           "tok": {d: s["tok.weight"][d].tolist() for d in (2, 3, 7, 8)}}, open("view.json", "w"))
print("ok")
