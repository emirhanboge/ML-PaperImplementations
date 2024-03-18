import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )  # First convolutional layer
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )  # Second convolutional layer, expansion is used to increase the number of channels
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # Here we define the layers of the ResNet model
        # These layers will be used to perform forward pass
        # Input: 3x224x224 image
        super(ResNet, self).__init__()
        self.inplanes = 64  # Number of input channels for the first convolutional layer
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # First convolutional layer
        self.bn1 = nn.BatchNorm2d(64)  # First batch normalization layer
        self.relu = nn.ReLU(inplace=True)  # ReLU activation function
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )  # Max pooling layer
        self.layer1 = self._make_layer(block, 64, layers[0])  # First residual block
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2
        )  # Second residual block
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2
        )  # Third residual block
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2
        )  # Fourth residual block
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Average pooling layer
        self.fc = nn.Linear(512, num_classes)  # Fully connected layer

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if (
            stride != 1 or self.inplanes != planes * block.expansion
        ):  # If the stride is not 1 or the number of input channels is not equal to the number of output channels
            # Then we need to downsample the input because the dimensions are not the same
            # Therefore, we need to use 1x1 convolutional layer to downsample the input
            downsample = nn.Sequential(  # 1x1 convolutional layer followed by batch normalization layer
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes)
            )  # Here the stride is always 1 and downsample is not needed

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # We pass the input image through the first residual block
        x = self.layer2(
            x
        )  # We pass the output of the first residual block through the second residual block
        x = self.layer3(
            x
        )  # We pass the output of the second residual block through the third residual block
        x = self.layer4(
            x
        )  # We pass the output of the third residual block through the fourth residual block

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(num_classes=1000):
    """Constructs a ResNet-18 model. BasicBlock is used because the number of layers is less than 50."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=1000):
    """Constructs a ResNet-34 model. BasicBlock is used because the number of layers is less than 50."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=1000):
    """Constructs a ResNet-50 model. Bottleneck is used because the number of layers is greater than 50."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=1000):
    """Constructs a ResNet-101 model. Bottleneck is used because the number of layers is greater than 50."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
