import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels: number of input channels
        out_channels: number of output channels

        Input Image: 64x64x3
        """
        super(UNet, self).__init__()

        # Contracting Path (Encoder)
        self.inc = self.double_conv(in_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 1024)

        # Expansive Path (Decoder)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Output Layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels, mid_channels=None):
        if not mid_channels:
            mid_channels = out_channels
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2), self.double_conv(in_channels, out_channels)
        )

    def upsampler(self, tensor, target_tensor):
        tensor = self.upsample(tensor)
        diff_y = target_tensor.size()[2] - tensor.size()[2]
        diff_x = target_tensor.size()[3] - tensor.size()[3]
        tensor = F.pad(
            tensor,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        tensor = torch.cat([target_tensor, tensor], 1)
        conv = self.double_conv(
            tensor.size()[1], target_tensor.size()[1], tensor.size()[1] // 2
        )
        out = conv(tensor)
        return out

    def forward(self, x):
        # Contracting Path (Encoder)
        inc = self.inc(x)
        down1 = self.down1(inc)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        # Expansive Path (Decoder)
        up1 = self.upsampler(down4, down3)
        up2 = self.upsampler(up1, down2)
        up3 = self.upsampler(up2, down1)
        up4 = self.upsampler(up3, inc)

        # Output Layer
        out = self.outc(up4)

        return out
