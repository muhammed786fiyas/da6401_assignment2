import torch
import torch.nn as nn
from models.vgg11 import VGG11


class PetClassifier(nn.Module):
    def __init__(self, num_classes=37, dropout_p=0.5):
        super(PetClassifier, self).__init__()
        self.model = VGG11(num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x):
        return self.model(x)

    def get_backbone(self):
        return self.model.features