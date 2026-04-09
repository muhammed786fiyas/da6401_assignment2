import torch
import torch.nn as nn
from models.layers import CustomDropout


class VGG11(nn.Module):
    def __init__(self, num_classes=37, dropout_p=0.3):
        super(VGG11, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(1024, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # better than normal init
                nn.init.constant_(m.bias, 0)

    def get_features(self):
        return self.features


class VGG11Encoder(nn.Module):
    def __init__(self, vgg11=None):
        super(VGG11Encoder, self).__init__()

        if vgg11 is None:
            vgg11 = VGG11()   

        blocks = list(vgg11.features.children())

        self.block1 = nn.Sequential(*blocks[0:4])
        self.block2 = nn.Sequential(*blocks[4:8])
        self.block3 = nn.Sequential(*blocks[8:15])
        self.block4 = nn.Sequential(*blocks[15:22])
        self.block5 = nn.Sequential(*blocks[22:])
        
    def forward(self, x):
        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        s5 = self.block5(s4)
        return s1, s2, s3, s4, s5