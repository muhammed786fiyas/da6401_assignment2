import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout


class PetLocalizer(nn.Module):
    def __init__(self, num_classes=37, dropout_p=0.3, freeze_backbone=False):
        super(PetLocalizer, self).__init__()

        vgg           = VGG11(num_classes=num_classes, dropout_p=dropout_p)
        self.features = vgg.features

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(256, 4),
        )

        self._init_regressor_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        # clamp to valid pixel range
        x = torch.clamp(x, min=0, max=224)
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

        # initialize final layer bias to predict image center
        # x_center=112, y_center=112, width=100, height=100
        final_layer = self.regressor[-1]
        nn.init.constant_(final_layer.bias, 0)
        nn.init.normal_(final_layer.weight, mean=0, std=0.01)
        with torch.no_grad():
            final_layer.bias.copy_(
                torch.tensor([112.0, 112.0, 100.0, 100.0])
            )