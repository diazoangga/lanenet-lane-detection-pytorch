import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
    def forward(self, x):
        return self.conv(x)

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dropout=dropout)
    def forward(self, x):
        return self.conv(x)

class down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, dropout=dropout)
        )
    def forward(self, x):
        return self.mpconv(x)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, dropout=0.1):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = double_conv(in_ch, out_ch, dropout=dropout)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_h = x1.size()[2] - x2.size()[2]
        diff_w = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, activation='sigmoid'):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, out_ch, kernel_size=1),
            nn.Sigmoid() if activation=='sigmoid' else nn.Softmax(dim=1)
            
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, dropout=0.1, bilinear=True):
        """
        Args:
            in_channels: number of input channels (default 3 for RGB)
            out_channels: number of segmentation classes (default 1 for binary segmentation)
            dropout: dropout probability in the conv blocks
            bilinear: whether to use bilinear upsampling (True) or ConvTranspose2d (False)
        """
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, 64, dropout=dropout)
        self.down1 = down(64, 128, dropout=dropout)
        self.down2 = down(128, 256, dropout=dropout)
        self.down3 = down(256, 512, dropout=dropout)
        self.down4 = down(512, 512, dropout=dropout)
        self.up1 = up(1024, 256, bilinear=bilinear, dropout=dropout)
        self.up2 = up(512, 128, bilinear=bilinear, dropout=dropout)
        self.up3 = up(256, 64, bilinear=bilinear, dropout=dropout)
        self.up4 = up(128, 64, bilinear=bilinear, dropout=dropout)
        self.outc_bin = outconv(64, 2, activation='softmax')
        self.outc_inst = outconv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)       # size: 64 x H x W
        x2 = self.down1(x1)    # size: 128 x H/2 x W/2
        x3 = self.down2(x2)    # size: 256 x H/4 x W/4
        x4 = self.down3(x3)    # size: 512 x H/8 x W/8
        x5 = self.down4(x4)    # size: 512 x H/16 x W/16
        x = self.up1(x5, x4)   # size: 256 x H/8 x W/8
        x = self.up2(x, x3)    # size: 128 x H/4 x W/4
        x = self.up3(x, x2)    # size: 64 x H/2 x W/2
        x = self.up4(x, x1)    # size: 64 x H x W
        output_x = self.outc_bin(x)  # size: out_channels x H x W

        y = self.up1(x5, x4)   # size: 256 x H/8 x W/8
        y = self.up2(y, x3)    # size: 128 x H/4 x W/4
        y = self.up3(y, x2)    # size: 64 x H/2 x W/2
        y = self.up4(y, x1)    # size: 64 x H x W
        output_y = self.outc_inst(y)  # size: out_channels x H x W
        return output_x, output_y
    
if __name__ == "__main__":   
    model = UNet()
    input_tensor = torch.randn(8, 3, 512, 256)
    output = model(input_tensor)
    print(output[0].shape, output[1].shape)