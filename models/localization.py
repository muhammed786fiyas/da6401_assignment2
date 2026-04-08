import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout


class PetLocalizer(nn.Module):
    def __init__(self, num_classes=37, dropout_p=0.3, freeze_backbone=False):
        super(PetLocalizer, self).__init__()

        # Backbone (same as classifier)
        vgg = VGG11(num_classes=num_classes, dropout_p=dropout_p)
        self.features = vgg.features

        # Freeze backbone if required
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Regression head (improved with BatchNorm)
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(1024, 4),
            nn.Sigmoid()
        )

        # Optional: initialize regressor weights (helps stability)
        self._init_regressor_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x

    def load_backbone(self, classifier_path):
        checkpoint = torch.load(classifier_path, map_location='cpu')

        backbone_state = {
            k.replace('model.features.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if 'model.features' in k
        }

        self.features.load_state_dict(backbone_state, strict=False)
        print("Backbone weights loaded into localizer.")

    def _init_regressor_weights(self):
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)