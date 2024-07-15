import torch
import torch.nn as nn
class CustomResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=952):
        super(CustomResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))  # Using LeakyReLU
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, groups=2)
        self.conv3_x = self._make_layer(block, 96, num_block[1], 2, kernel_size=5) 
        self.conv4_x = self._make_layer(block, 192, num_block[2], 2, groups=2)
        self.conv5_x = self._make_layer(block, 384, num_block[3], 2, kernel_size=5)  
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(384 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, groups=1, kernel_size=3):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, groups, kernel_size))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

class CustomBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, groups=1, kernel_size=3):
        super(CustomBlock, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),  # Using LeakyReLU
            nn.Conv2d(out_channels, out_channels * CustomBlock.expansion, kernel_size=kernel_size, padding=kernel_size//2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels * CustomBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != CustomBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * CustomBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * CustomBlock.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# Example usage
def custom_resnet18(num_classes=952):
    return CustomResNet(CustomBlock, [2, 2, 2, 2], num_classes=num_classes)
