import os
import torch

from pipeline.models.loaders import load_model


MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "vision_model_final.pt")
ONNX_PATH = os.path.join(MODEL_DIR, "vision_model.onnx")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model, brand_classes, model_classes, condition_classes = load_model(
    MODEL_PATH,
    MODEL_DIR,
    DEVICE
)

model.to(DEVICE)
model.eval()


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["brand_output"]   # důležité!

wrapped_model = WrappedModel(model).to(DEVICE)
wrapped_model.eval()


x = torch.randn(1, 3, 224, 224).to(DEVICE)


torch.onnx.export(
    wrapped_model,
    x,
    ONNX_PATH,
    input_names=["input"],
    output_names=["brand_output"],
    opset_version=11,
    do_constant_folding=True
)

print("ONNX model saved to:", ONNX_PATH)