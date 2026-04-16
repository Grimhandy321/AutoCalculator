import torch
import torch.nn as nn
from torchvision import models


class VisionModel(nn.Module):
    def __init__(self, num_brands, num_models, num_conditions):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.backbone.classifier = nn.Identity()

        self.shared = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.brand_output = nn.Linear(256, num_brands)
        self.model_output = nn.Linear(256, num_models)
        self.condition_output = nn.Linear(256, num_conditions)

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)

        return {
            "brand_output": self.brand_output(x),
            "model_output": self.model_output(x),
            "condition_output": self.condition_output(x),
        }


def load_vision_model(model_path, num_brands, num_models, num_conditions, device):
    model = VisionModel(num_brands, num_models, num_conditions).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model.eval()

    return model