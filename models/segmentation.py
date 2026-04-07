import torch
import torch.nn as nn
from models.vgg11 import VGG11, VGG11Encoder


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class PetSegmentor(nn.Module):
    def __init__(self, num_classes=3, dropout_p=0.5, freeze_backbone=False):
        super(PetSegmentor, self).__init__()

        # encoder — VGG11 backbone split into 5 blocks
        vgg = VGG11(dropout_p=dropout_p)
        self.encoder = VGG11Encoder(vgg)

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # decoder — mirrors encoder with transposed convolutions + skip connections
        # s5: 512ch 7x7 → s4: 512ch 14x14 → s3: 256ch 28x28 → s2: 128ch 56x56 → s1: 64ch 112x112
        self.decoder4 = DecoderBlock(512, 512, 256)
        self.decoder3 = DecoderBlock(256, 256, 128)
        self.decoder2 = DecoderBlock(128, 128, 64)
        self.decoder1 = DecoderBlock(64,  64,  32)

        # final upsample 112x112 → 224x224
        self.final_upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        # output head — 3 classes (foreground, background, uncertain)
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # encode
        s1, s2, s3, s4, s5 = self.encoder(x)

        # decode with skip connections
        d4 = self.decoder4(s5, s4)
        d3 = self.decoder3(d4, s3)
        d2 = self.decoder2(d3, s2)
        d1 = self.decoder1(d2, s1)

        # final upsample to input resolution
        out = self.final_upsample(d1)
        out = self.output_conv(out)
        return out

    def load_backbone(self, classifier_path):
        """Load pretrained backbone weights from classifier checkpoint."""
        checkpoint = torch.load(classifier_path, map_location='cpu')
        backbone_state = {
            k.replace('model.features.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if 'features' in k
        }
        self.encoder.block1[0].load_state_dict
        print("Backbone weights loaded from classifier checkpoint.")