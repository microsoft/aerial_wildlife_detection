'''
    Parts of the U-net model.

    Detectron2-compliant adaptation of:
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

    2022 Benjamin Kellenberger
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, SEM_SEG_HEADS_REGISTRY, Backbone, ShapeSpec


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



''' Detectron2-compliant wrappers '''
@BACKBONE_REGISTRY.register()
class UnetEncoder(Backbone):
    
    def __init__(self, cfg, input_shape):
        super(UnetEncoder, self).__init__()

        # params
        self.n_channels = cfg.MODEL.BACKBONE.get('NUM_CHANNELS', 3)     #TODO
        self.bilinear = cfg.MODEL.get('BILINEAR', True)    #TODO

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
    
    def forward(self, image):
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return {
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'x5': x5
        }
    
    def output_shape(self):
        # not needed since our UnetDecoder (below) is hard-coded, but here for
        # completeness
        return {
            'x1': ShapeSpec(channels=64, stride=1),
            'x2': ShapeSpec(channels=128, stride=2),
            'x3': ShapeSpec(channels=256, stride=2),
            'x4': ShapeSpec(channels=512, stride=2),
            'x5': ShapeSpec(channels=1024, stride=2)
        }


@SEM_SEG_HEADS_REGISTRY.register()
class UnetDecoder(nn.Module):

    def __init__(self, cfg, input_shape={}):
        super(UnetDecoder, self).__init__()

        self.n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.bilinear = cfg.MODEL.get('BILINEAR', True) #TODO
        factor = 2 if self.bilinear else 1
        self.ignore_value = -1       #TODO
        self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_value)

        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)
    
    def forward(self, features, targets=None):
        '''
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        '''
        x = self.up1(features['x5'], features['x4'])
        x = self.up2(x, features['x3'])
        x = self.up3(x, features['x2'])
        x = self.up4(x, features['x1'])
        y = self.outc(x)

        if self.training:
            return None, self.losses(y, targets)
        else:
            return y, {}
            # y = F.interpolate(
            #     y, scale_factor=self.common_stride, mode='bilinear', align_corners=False
            # )
            # return y, {}
    
    def losses(self, predictions, targets):
        # predictions = F.interpolate(
        #     predictions, scale_factor=self.common_stride, mode='bilinear', align_corners=False
        # )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss}
        return losses