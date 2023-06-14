import torch.nn as nn
from Sep_STS_Layer import SepSTSBasicLayer


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self, dim):
        super().__init__(
            nn.Conv3d(3, dim, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=False))


class SepSTSLayer(nn.Module):
    def __init__(self, plane, depth, num_frames, num_heads, window_size):
        super(SepSTSLayer, self).__init__()
        self.upper = SepSTSBasicLayer(plane, depth=depth, num_heads=num_heads,
                                 depth_window_size=window_size, point_window_size=(num_frames, 1, 1))

    def forward(self, x):
        out = self.upper(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, channel, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel, channel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)

        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)

        x += res
        return self.relu(x)

class ResLayer(nn.Module):
    def __init__(self, plane, num_layer, kernel_size=3):
        super(ResLayer, self).__init__()
        self.layer = nn.ModuleList()
        for i in range(num_layer):
            self.layer.append(ResBlock(plane, kernel_size=kernel_size))

        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)


class SepSTSEncoder(nn.Module):

    def __init__(self, nf, NF, window_size, nh):
        super(SepSTSEncoder, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=nf[-1]//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            ResBlock(nf[-1]//2, kernel_size=3),
        )

        self.stage1 = SepSTSLayer(nf[-1], depth=2, num_frames=NF, num_heads=nh[0], window_size=window_size[0])
        self.stage2 = SepSTSLayer(nf[-2], depth=2, num_frames=NF, num_heads=nh[1], window_size=window_size[1])
        self.stage3 = SepSTSLayer(nf[-3], depth=6, num_frames=NF, num_heads=nh[2], window_size=window_size[2])
        self.stage4 = SepSTSLayer(nf[-4], depth=2, num_frames=NF, num_heads=nh[3], window_size=window_size[3])

        self.down0 = nn.Conv3d(in_channels=nf[-1]//2, out_channels=nf[-1], kernel_size=(3,3,3), stride=(1,2,2), padding=1)
        self.down1 = nn.Conv3d(in_channels=nf[-1], out_channels=nf[-2], kernel_size=(3,3,3), stride=(1,2,2), padding=1)
        self.down2 = nn.Conv3d(in_channels=nf[-2], out_channels=nf[-3], kernel_size=(3,3,3), stride=(1,2,2), padding=1)
        self.down3 = nn.Conv3d(in_channels=nf[-3], out_channels=nf[-4], kernel_size=(3,3,3), stride=(1,2,2), padding=1)

    def forward(self, x):
        x0 = self.stem(x)

        x1 = self.down0(x0)
        x1 = self.stage1(x1)
        
        x2 = self.down1(x1)
        x2 = self.stage2(x2)

        x3 = self.down2(x2)
        x3 = self.stage3(x3)

        x4 = self.down3(x3)
        x4 = self.stage4(x4)

        return x0, x1, x2, x3, x4