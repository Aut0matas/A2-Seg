import torch.nn as nn
import torch
from math import sqrt

channel_dim = 3
ndf = 64


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class NetC(nn.Module):
    def __init__(self):
        super(NetC, self).__init__()
        self.convblock1 = ConvBlock(channel_dim, ndf, 7, 2, 3, bias=False)
        self.convblock2 = ConvBlock(ndf * 1, ndf * 2, 5, 2, 2, bias=False)
        self.convblock3 = ConvBlock(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.convblock4 = ConvBlock(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.convblock5 = ConvBlock(ndf * 8, ndf * 8, 4, 2, 1, bias=False)
        self.convblock6 = ConvBlock(ndf * 8, ndf * 8, 3, 2, 1, bias=False)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        batchsize = input.size()[0]
        out1 = self.convblock1(input)
        out2 = self.convblock2(out1)
        out3 = self.convblock3(out2)
        out4 = self.convblock4(out3)
        out5 = self.convblock5(out4)
        out6 = self.convblock6(out5)
        output = torch.cat(
            (
                input.view(batchsize, -1),
                1 * out1.view(batchsize, -1),
                2 * out2.view(batchsize, -1),
                2 * out3.view(batchsize, -1),
                2 * out4.view(batchsize, -1),
                2 * out5.view(batchsize, -1),
                4 * out6.view(batchsize, -1),
            ),
            1,
        )
        return output
