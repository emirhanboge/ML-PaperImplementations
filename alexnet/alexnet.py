import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        # Here we define the layers of the AlexNet model
        # These layers will be used to perform forward pass
        # Input: 3x224x224 image
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 96, kernel_size=11, stride=4, padding=2
        )  # 3 input channels, 96 output channels, 11x11 kernel size
        self.lrn1 = nn.LocalResponseNorm(
            size=5, alpha=0.0001, beta=0.75, k=2
        )  # Local Response Normalization
        self.conv2 = nn.Conv2d(
            96, 256, kernel_size=5, padding=2
        )  # 96 input channels, 256 output channels, 5x5 kernel size
        self.lrn2 = nn.LocalResponseNorm(
            size=5, alpha=0.0001, beta=0.75, k=2
        )  # Local Response Normalization
        self.conv3 = nn.Conv2d(
            256, 384, kernel_size=3, padding=1
        )  # 256 input channels, 384 output channels, 3x3 kernel size
        self.conv4 = nn.Conv2d(
            384, 384, kernel_size=3, padding=1
        )  # 384 input channels, 384 output channels, 3x3 kernel size
        self.conv5 = nn.Conv2d(
            384, 256, kernel_size=3, padding=1
        )  # 384 input channels, 256 output channels, 3x3 kernel size
        self.fc1 = nn.Linear(
            256 * 6 * 6, 4096
        )  # 256*6*6 input features, 4096 output features
        self.fc2 = nn.Linear(4096, 4096)  # 4096 input features, 4096 output features
        self.fc3 = nn.Linear(
            4096, num_classes
        )  # 4096 input features, num_classes output features
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2
        )  # 3x3 kernel size, 2 stride
        self.dropout = nn.Dropout(
            0.5
        )  # 50% dropout, this layer is used to prevent overfitting
        self.relu = nn.ReLU()  # ReLU activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax activation function

    def forward(self, x):
        # Here we define the forward pass of the AlexNet model
        # The forward pass is the process of inputting data to the model and computing the output
        x = self.relu(self.conv1(x))
        x = self.lrn1(x)
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.lrn2(x)
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
