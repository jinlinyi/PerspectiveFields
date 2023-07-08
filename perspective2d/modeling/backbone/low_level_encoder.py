import torch.nn as nn


class LowLevelEncoder(nn.Module):
    def __init__(self, feat_dim=64, in_channel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, feat_dim, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(feat_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
