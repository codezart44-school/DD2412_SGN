# Archtitechture : Wide Resnet 28 2
import torch                    # core lib for tensors and computations
import torch.nn as nn           # provides layers like Linear, Conv2d, BatchNorm2d etc.
import torch.functional as F    # provides functions for activation, loss etc.
# import torch.optim as optim     # provides optimizers like SGD, Adam, etc for training. 


class ResnetBlock(nn.Module):
    """Basic residual block of two 3x3 convolutions."""
    def __init__(self, in_channels, out_channels, stride=1, conv_l2=0.0, bn_l2=0.0, version=1):
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
        if self.version == 2: # Version changes the order of the relu activation 
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
        else:
            shortcut = self.shortcut(x)
            out = F.relu(self.bn1(x))
        
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        if self.version == 1:
            out = self.bn2(out)
        
        out += shortcut
        if self.version == 1:
            out = F.relu(out)

        return out


class WideResNet(nn.Module):
    """Wide Residual Network."""
    def __init__(self, depth, width_multiplier, num_classes=10, input_shape=(3, 32, 32), version=1, l2=0.0):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4."
        num_blocks = (depth - 4) // 6

        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_group(16 * width_multiplier, num_blocks, stride=1, version=version, conv_l2=l2, bn_l2=l2)
        self.layer2 = self._make_group(32 * width_multiplier, num_blocks, stride=2, version=version, conv_l2=l2, bn_l2=l2)
        self.layer3 = self._make_group(64 * width_multiplier, num_blocks, stride=2, version=version, conv_l2=l2, bn_l2=l2)

        self.bn = nn.BatchNorm2d(64 * width_multiplier, eps=1e-5, momentum=0.9)
        self.fc = nn.Linear(64 * width_multiplier, num_classes)

    def _make_group(self, out_channels, num_blocks, stride, version, conv_l2, bn_l2):
        layers = []
        layers.append(ResnetBlock(self.in_channels, out_channels, stride, conv_l2, bn_l2, version))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResnetBlock(out_channels, out_channels, stride=1, conv_l2=conv_l2, bn_l2=bn_l2, version=version))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_model(depth=28, width_multiplier=2, num_classes=10, input_shape=(3, 32, 32), version=1, l2=0.0):
    """Create the WideResNet model."""
    return WideResNet(depth, width_multiplier, num_classes, input_shape, version, l2)


# Example usage
model = create_model(depth=28, width_multiplier=2, num_classes=10, version=1, l2=0.0)
print(model)
