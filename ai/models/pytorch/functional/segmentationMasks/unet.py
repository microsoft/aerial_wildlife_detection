'''
    U-Net implementation in PyTorch (Ronneberger et al., 2015).

    Adapted from 2019 Joris van Vugt (source: https://raw.githubusercontent.com/jvanvugt/pytorch-unet/master/unet.py).
    2020 Benjamin Kellenberger
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, labelclassmap, in_channels=3, depth=5, numFeaturesExponent=6, padding=False, batch_norm=False, upsamplingMode='upconv'):
        super(UNet, self).__init__()
        self.labelclassmap = labelclassmap
        self.in_channels = in_channels
        self.depth = depth
        self.numFeaturesExponent = numFeaturesExponent
        self.padding = padding
        assert upsamplingMode in ('upconv', 'upsample')
        self.upsamplingMode = upsamplingMode

        wf = self.numFeaturesExponent
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), upsamplingMode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, len(self.labelclassmap.keys()), kernel_size=1)


    def getStateDict(self):
        stateDict = {
            'model_state': self.state_dict(),
            'labelclassMap': self.labelclassMap,
            'in_channels': self.in_channels,
            'depth': self.depth,
            'numFeaturesExponent': self.numFeaturesExponent,
            'padding': self.padding,
            'upsamplingMode': self.upsamplingMode
        }
        return stateDict


    @staticmethod
    def loadFromStateDict(stateDict):
        # parse args
        labelclassMap = stateDict['labelclassMap']
        in_channels = (stateDict['in_channels'] if 'in_channels' in stateDict else 3)
        depth = (stateDict['depth'] if 'depth' in stateDict else 5)
        numFeaturesExponent = (stateDict['numFeaturesExponent'] if 'numFeaturesExponent' in stateDict else 6)
        padding = (stateDict['padding'] if 'padding' in stateDict else False)
        batch_norm = (stateDict['batch_norm'] if 'batch_norm' in stateDict else False)
        upsamplingMode = (stateDict['upsamplingMode'] if 'upsamplingMode' in stateDict else 'upconv')
        state = (stateDict['model_state'] if 'model_state' in stateDict else None)

        # return model
        model = UNet(labelclassMap, in_channels, depth, numFeaturesExponent, padding, batch_norm, upsamplingMode)
        if state is not None:
            model.load_state_dict(state)
        return model


    @staticmethod
    def averageStateDicts(stateDicts):
        model = UNet.loadFromStateDict(stateDicts[0])
        pars = dict(model.named_parameters())
        for key in pars:
            pars[key] = pars[key].detach().cpu()
        for s in range(1,len(stateDicts)):
            nextModel = UNet.loadFromStateDict(stateDicts[s])
            state = dict(nextModel.named_parameters())
            for key in state:
                pars[key] += state[key].detach().cpu()
        finalState = stateDicts[-1]
        for key in pars:
            finalState['model_state'][key] = pars[key] / (len(stateDicts))

        return finalState


    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)



class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)


    def forward(self, x):
        out = self.block(x)
        return out



class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)


    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]


    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out