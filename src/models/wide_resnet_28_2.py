# Archtitechture : Wide Resnet 28 2
import torch                    # core lib for tensors and computations
import torch.nn as nn           # provides layers like Linear, Conv2d, BatchNorm2d etc.
import torch.nn.functional as F    # provides functions for activation, loss etc.
# import torch.optim as optim     # provides optimizers like SGD, Adam, etc for training. 

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, version=1):

        """
        Args:
          `in_channels`:
          `out_channels`:
          `stride`:
          `version`:
        """
        super().__init__()
        self.version = version

        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        if self.version == 2:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
        else:
            shortcut = self.shortcut(x)
            out = F.relu(self.bn1(x))

        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += shortcut

        if self.version == 1:
            out = F.relu(out)

        return out

class WideResNet(nn.Module):
    def __init__(self, depth, width_multiplier, num_classes=10, input_shape=(3, 32, 32), version=1, sgn=False):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4."
        num_blocks = (depth - 4) // 6

        self.sgn = sgn

        self.in_channels = 16
        self.conv1 = nn.Conv2d(input_shape[0], self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_group(16 * width_multiplier, num_blocks, stride=1, version=version)
        self.layer2 = self._make_group(32 * width_multiplier, num_blocks, stride=2, version=version)
        self.layer3 = self._make_group(64 * width_multiplier, num_blocks, stride=2, version=version)

        self.bn = nn.BatchNorm2d(64 * width_multiplier, eps=1e-5, momentum=0.9)

        if not sgn:
            self.fc = nn.Linear(64 * width_multiplier, num_classes)
        else:
            self.fc_mu = nn.Linear(64 * width_multiplier, num_classes - 1)
            self.fc_r = nn.Linear(64 * width_multiplier, num_classes - 1)

    def _make_group(self, out_channels, num_blocks, stride, version):
        layers = []
        layers.append(ResnetBlock(self.in_channels, out_channels, stride, version))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResnetBlock(out_channels, out_channels, stride=1, version=version))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
        x = torch.flatten(x, 1)

        if not self.sgn:
            x = self.fc(x)
            return x
        else:
            mu = self.fc_mu(x)
            r = self.fc_r(x)
            return mu, r

def create_model(depth=28, width_multiplier=2, num_classes=10, input_shape=(3, 32, 32), version=1, sgn=False):
    """Create the WideResNet model with no manual L2 penalty."""
    return WideResNet(depth, width_multiplier, num_classes, input_shape, version, sgn=False)

def main():
    # Example usage
    model = create_model(depth=28, width_multiplier=2, num_classes=10, version=1)
    print(model)


if __name__=='__main__':
    main()
