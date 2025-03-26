import torch
import torch.nn as nn
import torch.nn.functional as F

# Two consecutive conv layers with BN, ReLU and dropout.
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

# A simple 1x1 convolutional block for final output.
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=None):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.activation = activation
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.act(x)
        return x

# UNet++ with nested skip connections.
class UNetPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, inst_channels=4, dropout=0.1, bilinear=True):
        """
        Args:
            in_channels: number of input channels (e.g. 3 for RGB)
            out_channels: number of segmentation classes
            inst_channels: number of instance segmentation channels
            dropout: dropout probability in conv blocks
            bilinear: if True, use bilinear upsampling (otherwise ConvTranspose2d)
        """
        super(UNetPP, self).__init__()
        nb_filter = [64, 128, 256, 512, 512]  # feature map sizes per level
        
        # Encoder (same as UNet)
        self.conv0_0 = double_conv(in_channels, nb_filter[0], dropout)
        self.conv1_0 = double_conv(nb_filter[0], nb_filter[1], dropout)
        self.conv2_0 = double_conv(nb_filter[1], nb_filter[2], dropout)
        self.conv3_0 = double_conv(nb_filter[2], nb_filter[3], dropout)
        self.conv4_0 = double_conv(nb_filter[3], nb_filter[4], dropout)
        
        # We'll use bilinear upsampling throughout.
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # First nested level:
        self.conv0_1 = double_conv(nb_filter[0] + nb_filter[1], nb_filter[0], dropout)
        self.conv1_1 = double_conv(nb_filter[1] + nb_filter[2], nb_filter[1], dropout)
        self.conv2_1 = double_conv(nb_filter[2] + nb_filter[3], nb_filter[2], dropout)
        self.conv3_1 = double_conv(nb_filter[3] + nb_filter[4], nb_filter[3], dropout)
        
        # Second nested level:
        self.conv0_2 = double_conv(nb_filter[0]*2 + nb_filter[1], nb_filter[0], dropout)
        self.conv1_2 = double_conv(nb_filter[1]*2 + nb_filter[2], nb_filter[1], dropout)
        self.conv2_2 = double_conv(nb_filter[2]*2 + nb_filter[3], nb_filter[2], dropout)
        
        # Third nested level:
        self.conv0_3 = double_conv(nb_filter[0]*3 + nb_filter[1], nb_filter[0], dropout)
        self.conv1_3 = double_conv(nb_filter[1]*3 + nb_filter[2], nb_filter[1], dropout)
        
        # Fourth nested level (final output from level 0 branch):
        # The input channels here are: x0_0, x0_1, x0_2, x0_3 and upsampled x1_3
        in_ch_out = nb_filter[0]*4 + nb_filter[1]  # 64*4 + 128 = 384
        
        # Binary segmentation head - no activation to output raw logits
        self.outc_bin = outconv(in_ch_out, out_channels, activation=None)
        
        # Instance segmentation head - no activation to output raw logits 
        self.outc_inst = outconv(in_ch_out, inst_channels, activation=None)
        
    def forward(self, x):
        # Encoder path.
        x0_0 = self.conv0_0(x)                     # size: [B, 64, H, W]
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))   # [B, 128, H/2, W/2]
        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))   # [B, 256, H/4, W/4]
        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))   # [B, 512, H/8, W/8]
        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))   # [B, 512, H/16, W/16]
        
        # First nested level.
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))  # [B, 64, H, W]
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))  # [B, 128, H/2, W/2]
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))  # [B, 256, H/4, W/4]
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))  # [B, 512, H/8, W/8]
        
        # Second nested level.
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))  # [B, 64, H, W]
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))  # [B, 128, H/2, W/2]
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))  # [B, 256, H/4, W/4]
        
        # Third nested level.
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))  # [B, 64, H, W]
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))  # [B, 128, H/2, W/2]
        
        # Feature concatenation for the final outputs
        final_features = torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1)
        
        # Generate both binary and instance segmentation outputs
        logits_bin = self.outc_bin(final_features)  # Binary segmentation logits
        logits_inst = self.outc_inst(final_features)  # Instance segmentation logits
        
        return logits_bin, logits_inst

if __name__ == '__main__':
    from torchsummary import summary
    # Test the UNet++ with a dummy input.
    datatype = torch.float8_e4m3fn
    model = UNetPP(in_channels=3, out_channels=2, inst_channels=4, dropout=0.1, bilinear=True).to(device='cuda')
    # summary(model, torch.Tensor(size=(3,64, 128)).half())
    print('model ok')
    input_tensor = torch.randn(2, 3, 512, 256).to('cuda')  # for example, a batch of 8 images
    bin_output, inst_output = model(input_tensor)
    print(f"Binary output shape: {bin_output.shape}")  # Expected: [8, 2, 512, 256]
    print(f"Instance output shape: {inst_output.shape}")  # Expected: [8, 4, 512, 256]