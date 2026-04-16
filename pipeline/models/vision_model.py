import torch
import torch.nn as nn
from torchvision import models


class VisionModel(nn.Module):
    def __init__(self, num_brands, num_models, num_conditions):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()

        self.shared = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.brand_head = nn.Linear(256, num_brands)
        self.model_head = nn.Linear(256, num_models)
        self.condition_head = nn.Linear(256, num_conditions)

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)

        return (
            self.brand_head(x),
            self.model_head(x),
            self.condition_head(x),
        )


def load_vision_model(model_path, num_brands, num_models, num_conditions, device):
    model = VisionModel(num_brands, num_models, num_conditions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model