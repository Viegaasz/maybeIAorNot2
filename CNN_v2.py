import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_shortcut=True):
        super(Bottleneck, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2
        self.use_shortcut = use_shortcut
        self.conv1 = Conv(in_channels, mid_channels, 1, 1, 0)
        self.conv2 = Conv(mid_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Sequential()
        if self.use_shortcut and in_channels != out_channels:
            self.shortcut = Conv(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_shortcut:
            residual = self.shortcut(residual)
        x += residual
        return x

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, mid_channels=None):
        super(C2f, self).__init__()
        self.conv = Conv(in_channels, out_channels, 3, 2, 1) # Downsample
        blocks = []
        for _ in range(num_blocks):
            blocks.append(Bottleneck(out_channels, out_channels, mid_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        return x

class Classify(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classify, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0)  # Evitar overffiting
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=8):
        super(ClassificationModel, self).__init__()
        self.model = nn.Sequential(
            Conv(3, 80, 3, 2, 1),
            Conv(80, 160, 3, 2, 1),
            C2f(160, 160, 3, 80),
            Conv(160, 320, 3, 2, 1),
            C2f(320, 320, 6, 160),
            Conv(320, 640, 3, 2, 1),
            C2f(640, 640, 6, 320),
            Conv(640, 1280, 3, 2, 1),
            C2f(1280, 1280, 3, 640),
            Classify(1280, num_classes)
        )

    def forward(self, x):
        return self.model(x)