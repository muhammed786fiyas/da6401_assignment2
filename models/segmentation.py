import torch
import torch.nn as nn
from models.vgg11 import VGG11, VGG11Encoder
from models.layers import CustomDropout


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

        vgg          = VGG11(dropout_p=dropout_p)
        self.encoder = VGG11Encoder(vgg)

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # decoder mirrors encoder
        # s5: 512ch 7x7   → decoder4 → 256ch 14x14
        # s4: 512ch 14x14 → decoder3 → 128ch 28x28
        # s3: 256ch 28x28 → decoder2 → 64ch  56x56
        # s2: 128ch 56x56 → decoder1 → 32ch  112x112
        # s1: 64ch  112x112
        self.decoder4 = DecoderBlock(512, 512, 256)
        self.decoder3 = DecoderBlock(256, 256, 128)
        self.decoder2 = DecoderBlock(128, 128, 64)
        self.decoder1 = DecoderBlock(64,  64,  32)

        # final upsample 112x112 → 224x224
        self.final_upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        # output head
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        # dropout in bottleneck
        self.bottleneck_dropout = CustomDropout(p=dropout_p)

    def forward(self, x):
        # encode
        s1, s2, s3, s4, s5 = self.encoder(x)

        # dropout at bottleneck
        s5 = self.bottleneck_dropout(s5)

        # decode with skip connections
        d4  = self.decoder4(s5, s4)
        d3  = self.decoder3(d4, s3)
        d2  = self.decoder2(d3, s2)
        d1  = self.decoder1(d2, s1)

        # final upsample to input resolution
        out = self.final_upsample(d1)
        out = self.output_conv(out)
        return out

    def load_backbone(self, classifier_path):
        checkpoint = torch.load(classifier_path, map_location='cpu')

        # extract only feature weights from classifier checkpoint
        backbone_state = {
            k.replace('model.features.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if 'model.features' in k
        }

        # load into a temp VGG11 then build encoder from it
        vgg_temp = VGG11()
        vgg_temp.features.load_state_dict(backbone_state, strict=False)

        encoder_temp = VGG11Encoder(vgg_temp)
        self.encoder.load_state_dict(encoder_temp.state_dict())
        print("Backbone weights loaded into segmentor encoder.")