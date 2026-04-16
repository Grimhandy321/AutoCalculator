import os
import torch
from torchviz import make_dot

from pipeline.models.loaders import load_model

MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "vision_model_final.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, brand_classes, model_classes, condition_classes = load_model(
    MODEL_PATH,
    MODEL_DIR,
    DEVICE
)

model.eval()


x = torch.randn(1, 3, 224, 224).to(DEVICE)


# FORWARD PASS
outputs = model(x)

graph_output = outputs["brand_output"]


# CREATE GRAPH
dot = make_dot(
    graph_output,
    params=dict(model.named_parameters())
)

out_file = os.path.join(MODEL_DIR, "vision_model_graph")
dot.render(out_file, format="png", cleanup=True)

print("Saved model graph to:", out_file + ".png")