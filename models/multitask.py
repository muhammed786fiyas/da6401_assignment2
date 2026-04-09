import os
import torch
import torch.nn as nn

from models.vgg11 import VGG11, VGG11Encoder
from models.layers import CustomDropout
from models.classification import PetClassifier
from models.localization import PetLocalizer
from models.segmentation import PetSegmentor, DecoderBlock


class MultiTaskPerceptionModel(nn.Module):
    def __init__(
        self,
        classifier_path = 'checkpoints/classifier.pth',
        localizer_path  = 'checkpoints/localizer.pth',
        unet_path       = 'checkpoints/unet.pth',
        num_classes     = 37,
        dropout_p       = 0.3,
    ):
        super(MultiTaskPerceptionModel, self).__init__()

        # ── download checkpoints from Google Drive ──
        import gdown
        gdown.download(id="1aJ9_EkH8ZT1Eq_B5lL2cuw1GESTb75aI", output=classifier_path, quiet=False)
        gdown.download(id="13dakOkqhhocmrSHUrXBxmFbyAh6fMBE8",  output=localizer_path,  quiet=False)
        gdown.download(id="1gGxfOgyoeANrvybKIDhHrovunbM1GW7Z",       output=unet_path,       quiet=False)

        # ── shared backbone ──
        vgg             = VGG11(num_classes=num_classes, dropout_p=dropout_p)
        self.encoder    = VGG11Encoder(vgg)

        # ── classification head ──
        self.cls_pool       = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_head       = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

        # ── localization head ──
        self.loc_head = nn.Sequential(
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

        # ── segmentation decoder ──
        self.decoder4       = DecoderBlock(512, 512, 256)
        self.decoder3       = DecoderBlock(256, 256, 128)
        self.decoder2       = DecoderBlock(128, 128, 64)
        self.decoder1       = DecoderBlock(64,  64,  32)
        self.final_upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.output_conv    = nn.Conv2d(32, 3, kernel_size=1)
        self.bottleneck_dropout = CustomDropout(p=dropout_p)

        # ── load weights from checkpoints ──
        self._load_weights(classifier_path, localizer_path, unet_path)

    def _load_weights(self, classifier_path, localizer_path, unet_path):
        device = torch.device('cpu')

        # ── load classifier weights → backbone + cls_head ──
        if os.path.exists(classifier_path):
            clf_ckpt = torch.load(classifier_path, map_location=device)
            clf_state = clf_ckpt['model_state_dict']

            # backbone from classifier
            backbone_state = {
                k.replace('model.features.', ''): v
                for k, v in clf_state.items()
                if 'model.features' in k
            }
            vgg_temp = VGG11()
            vgg_temp.features.load_state_dict(backbone_state, strict=False)
            encoder_temp = VGG11Encoder(vgg_temp)
            self.encoder.load_state_dict(encoder_temp.state_dict())

            # cls_head from classifier
            cls_head_state = {
                k.replace('model.classifier.', ''): v
                for k, v in clf_state.items()
                if 'model.classifier' in k
            }
            self.cls_head.load_state_dict(cls_head_state, strict=False)
            print("Classifier weights loaded.")

        # ── load localizer weights → loc_head ──
        if os.path.exists(localizer_path):
            loc_ckpt   = torch.load(localizer_path, map_location=device)
            loc_state  = loc_ckpt['model_state_dict']

            loc_head_state = {
                k.replace('regressor.', ''): v
                for k, v in loc_state.items()
                if k.startswith('regressor.')
            }
            self.loc_head.load_state_dict(loc_head_state, strict=False)
            print("Localizer weights loaded.")

        # ── load unet weights → segmentation decoder ──
        if os.path.exists(unet_path):
            seg_ckpt  = torch.load(unet_path, map_location=device)
            seg_state = seg_ckpt['model_state_dict']

            # decoder blocks
            for name in ['decoder4', 'decoder3', 'decoder2', 'decoder1',
                         'final_upsample', 'output_conv', 'bottleneck_dropout']:
                block_state = {
                    k.replace(f'{name}.', ''): v
                    for k, v in seg_state.items()
                    if k.startswith(f'{name}.')
                }
                if block_state:
                    getattr(self, name).load_state_dict(block_state, strict=False)

            print("Segmentor weights loaded.")

    def forward(self, x):
        # single forward pass through shared encoder
        s1, s2, s3, s4, s5 = self.encoder(x)

        # ── classification ──
        cls_feat = torch.flatten(s5, 1)
        cls_out  = self.cls_head(cls_feat)

        # ── localization ──
        loc_out  = self.loc_head(s5)
        loc_out  = torch.clamp(loc_out, min=0, max=224)

        # ── segmentation ──
        s5_drop  = self.bottleneck_dropout(s5)
        d4       = self.decoder4(s5_drop, s4)
        d3       = self.decoder3(d4, s3)
        d2       = self.decoder2(d3, s2)
        d1       = self.decoder1(d2, s1)
        seg_out  = self.final_upsample(d1)
        seg_out  = self.output_conv(seg_out)

        return cls_out, loc_out, seg_out