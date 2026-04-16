import torch
import numpy as np
from pipeline.models.vision_model import VisionModel


def load_model(model_path, model_dir, device):
    brand_classes = np.load(f"{model_dir}/brand_classes.npy", allow_pickle=True)
    model_classes = np.load(f"{model_dir}/model_classes.npy", allow_pickle=True)
    cond_classes = np.load(f"{model_dir}/condition_classes.npy", allow_pickle=True)

    model = VisionModel(
        len(brand_classes),
        len(model_classes),
        len(cond_classes)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, brand_classes, model_classes, cond_classes