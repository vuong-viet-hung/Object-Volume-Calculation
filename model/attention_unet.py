import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down_conv(x)


class UpConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.UpConvBlock = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.UpConvBlock(x)


class AttentionBlock(torch.nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = torch.nn.Sequential(
            torch.nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(f_int)
        )

        self.w_x = torch.nn.Sequential(
            torch.nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(f_int)
        )

        self.psi = torch.nn.Sequential(
            torch.nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid()
        )

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_1 = self.w_g(g)
        x_1 = self.w_x(x)
        psi = self.relu(g_1 + x_1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_5 = ConvBlock(in_channels=512, out_channels=1024)

        self.up_5 = UpConvBlock(in_channels=1024, out_channels=512)
        self.att_5 = AttentionBlock(f_g=512, f_l=512, f_int=256)
        self.upconv_5 = ConvBlock(in_channels=1024, out_channels=512)

        self.up_4 = UpConvBlock(in_channels=512, out_channels=256)
        self.att_4 = AttentionBlock(f_g=256, f_l=256, f_int=128)
        self.upconv_4 = ConvBlock(in_channels=512, out_channels=256)

        self.up_3 = UpConvBlock(in_channels=256, out_channels=128)
        self.att_3 = AttentionBlock(f_g=128, f_l=128, f_int=64)
        self.upconv_3 = ConvBlock(in_channels=256, out_channels=128)

        self.up_2 = UpConvBlock(in_channels=128, out_channels=64)
        self.att_2 = AttentionBlock(f_g=64, f_l=64, f_int=32)
        self.upconv_2 = ConvBlock(in_channels=128, out_channels=64)

        self.conv_1x1 = torch.nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoding path
        x_1 = self.conv_1(x)

        x_2 = self.maxpool(x_1)
        x_2 = self.conv_2(x_2)

        x_3 = self.maxpool(x_2)
        x_3 = self.conv_3(x_3)

        x_4 = self.maxpool(x_3)
        x_4 = self.conv_4(x_4)

        x_5 = self.maxpool(x_4)
        x_5 = self.conv_5(x_5)

        # Decoding & concatenating path
        d_5 = self.up_5(x_5)
        x_4 = self.att_5(g=d_5, x=x_4)
        d_5 = torch.cat((x_4, d_5), dim=1)
        d_5 = self.upconv_5(d_5)

        d_4 = self.up_4(d_5)
        x_3 = self.att_4(g=d_4, x=x_3)
        d_4 = torch.cat((x_3, d_4), dim=1)
        d_4 = self.upconv_4(d_4)

        d_3 = self.up_3(d_4)
        x_2 = self.att_3(g=d_3, x=x_2)
        d_3 = torch.cat((x_2, d_3), dim=1)
        d_3 = self.upconv_3(d_3)

        d_2 = self.up_2(d_3)
        x_1 = self.att_2(g=d_2, x=x_1)
        d_2 = torch.cat((x_1, d_2), dim=1)
        d_2 = self.upconv_2(d_2)

        d_1 = self.conv_1x1(d_2)
        return d_1
