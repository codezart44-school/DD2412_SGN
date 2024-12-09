# Archtitechture : Wide Resnet 28 2
import torch                    # core lib for tensors and computations
import torch.nn as nn           # provides layers like Linear, Conv2d, BatchNorm2d etc.
import torch.nn.functional as F    # provides functions for activation, loss etc.
# import torch.optim as optim     # provides optimizers like SGD, Adam, etc for training. 

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, conv_l2=0.0, bn_l2=0.0, version=1):
        super().__init__()
        self.version = version
        self.conv_l2 = conv_l2
        self.bn_l2 = bn_l2

        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=0.3)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        l2_penalty = 0.0

        if self.version == 2:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
        else:
            shortcut = self.shortcut(x)
            out = F.relu(self.bn1(x))

        out = self.conv1(out)
        l2_penalty += self._compute_l2_penalty(self.conv1, self.conv_l2)

        out = self.bn2(out)
        l2_penalty += self._compute_l2_penalty(self.bn2, self.bn_l2)

        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        l2_penalty += self._compute_l2_penalty(self.conv2, self.conv_l2)

        out += shortcut
        if self.version == 1:
            out = F.relu(out)

        return out, l2_penalty

    def _compute_l2_penalty(self, layer, l2_factor):
        if hasattr(layer, 'weight') and layer.weight.requires_grad and l2_factor > 0.0:
            return l2_factor * layer.weight.pow(2).sum()
        return 0.0


class WideResNet(nn.Module):
    def __init__(self, depth, width_multiplier, num_classes=10, input_shape=(3, 32, 32), version=1, l2=0.0):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4."
        num_blocks = (depth - 4) // 6

        self.in_channels = 16
        self.conv1 = nn.Conv2d(input_shape[0], self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)

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
        total_l2_penalty = 0.0

        x = self.conv1(x)
        x, l2_penalty = self._apply_block(self.layer1, x)
        total_l2_penalty += l2_penalty

        x, l2_penalty = self._apply_block(self.layer2, x)
        total_l2_penalty += l2_penalty

        x, l2_penalty = self._apply_block(self.layer3, x)
        total_l2_penalty += l2_penalty

        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, total_l2_penalty

    def _apply_block(self, block, x):
        total_l2_penalty = 0.0
        for layer in block:
            x, l2_penalty = layer(x)
            total_l2_penalty += l2_penalty
        return x, total_l2_penalty


def create_model(depth=28, width_multiplier=2, num_classes=10, input_shape=(3, 32, 32), version=1, l2=0.0):
    """Create the WideResNet model."""
    return WideResNet(depth, width_multiplier, num_classes, input_shape, version, l2)



def main():
    # Example usage
    model = create_model(depth=28, width_multiplier=2, num_classes=10, version=1, l2=0.0)
    print(model)


if __name__=='__main__':
    main()
