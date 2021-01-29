import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.utils import load_state_dict_from_url

from kronos.models.layers import conv2d, SpatialSoftArgmax


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 64)

        self.num_ctx_frames = 1

    def forward(self, x):
        batch_size, t, c, h, w = x.shape
        x = x.view((batch_size * t, c, h, w))
        feats = self.model(x)
        return {
            "embs": feats.view((batch_size, t, -1)),
        }


class SiameseAENet(ResNet):
    """Siamese net with a reconstruction loss."""

    def __init__(
        self, block=BasicBlock, layers=[2, 2, 2, 2], **kwargs):
        super().__init__(block, layers, **kwargs)

        # Load pretrained weights.
        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            progress=True)
        self.load_state_dict(state_dict)

        # Embedding head.
        self.fc = nn.Linear(self.fc.in_features, 64)

        # Upsampling path.
        # self.up1 = Up(1024, 512 // 2)
        # self.up2 = Up(512, 256 // 2)
        # self.up3 = Up(256, 128 // 2)
        # self.up4 = Up(128, 64)
        self.up_convs = nn.ModuleList([
            conv2d(512, 256),
            conv2d(256, 128),
            conv2d(128, 64),
        ])
        # self.up1 = conv2d(512, 256)
        # self.up2 = conv2d(256, 128)
        # self.up3 = conv2d(128, 64)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

        self.num_ctx_frames = 1

    def encode(self, x):
        # Compute embeddings.
        batch_size, t, c, h, w = x.shape
        x = x.view((batch_size * t, c, h, w))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)   # B, 64, 56, 56
        x2 = self.layer2(x1)  # B, 128, 28, 28
        x3 = self.layer3(x2)  # B, 256, 14, 14
        x4 = self.layer4(x3)  # B, 512, 7, 7

        # Compute embeddings.
        feats = self.avgpool(x4)  # B, 512, 1, 1
        flat_feats = torch.flatten(feats, 1)
        # embs = self.fc(flat_feats)
        embs = embs.view((batch_size, t, -1))

        return embs, [x1, x2, x3, x4, feats]

    def decode_all_res(self, feature_maps):
        """Decode using all spatial resolutions, a la u-net."""
        x1, x2, x3, x4, feats = feature_maps
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        recon = self.out_conv(x)
        return recon

    def decode_lowest_res(self, feature_maps):
        _, _, _, x, _ = feature_maps
        for up_conv in self.up_convs:
            x = F.relu(up_conv(x))
            x = F.interpolate(
                x,
                scale_factor=2,
                mode='bilinear',
                recompute_scale_factor=False,
                align_corners=True)
        x = self.out_conv(x)
        return x

    def forward(self, x, reconstruct=True):
        embs, feature_maps = self.encode(x)
        ret = {"embs": embs}
        if reconstruct:
            ret['reconstruction'] = self.decode_lowest_res(feature_maps)
        return ret


class SupervisedNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 128)
        self.head = nn.Linear(128, out_dim)

        self.num_ctx_frames = 1

    def forward(self, x):
        return self.head(F.relu(self.model(x)))
