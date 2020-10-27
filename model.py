import math
import os, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torchvision import transforms, datasets

"""
Borrowd from https://github.com/thstkdgus35/EDSR-PyTorch
Used for upsampling
"""
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSR(nn.Module):
    def __init__(self, num_layers=32, feature_size=256, scale=2, output_channels=3):
        print("Building EDSR...")
        super(EDSR, self).__init__()

        self.scaling_factor = 0.1
        self.feature_size = feature_size
        self.num_layers = num_layers

        self.conv1 = self._conv3x3(3, feature_size)
        self.resBlock = self._resBlock(feature_size, feature_size)
        self.conv2 = self._conv3x3(feature_size, feature_size)
        self.tail = [
            Upsampler(self._conv3x3, scale=scale, n_feats=feature_size),
            self._conv3x3(feature_size, output_channels)
        ]

        self.upsample = nn.Sequential(*self.tail)


    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def _resBlock(self, x, planes, stride=1):
        return nn.Sequential(
            nn.Conv2d(x, planes, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                      padding=1, bias=False),
        )

    def forward(self, input):
        x = self.conv1(input-0.5)
        conv_1 = x

        for i in range(self.num_layers):
            x = self.resBlock(x) * self.scaling_factor

        x = self.conv2(x)
        x += conv_1

        x = self.upsample(x)
        return x, torch.clamp(x+0.5, 0, 1.0)

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    input = torch.autograd.Variable(torch.randn(1, 3, 100, 100))  # batch, ch, h, w
    label = torch.autograd.Variable(torch.randn(1, 3, 200, 200))  # batch, ch, h, w

    # input = input.to(device)

    model = EDSR(num_layers=8, feature_size=32).train()
    # output = model(input)
    # print(output.shape)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    output = model(input)
    loss = F.l1_loss(output, label)
    loss.backward()
    optimizer.step()