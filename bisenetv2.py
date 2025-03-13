import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, strides, padding, activation=True):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size, strides, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        if activation:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

    def forward(self, input_tensors):
        # print('Input Tensors: ', input_tensors.shape)
        conv = self.conv2d(input_tensors)
        bn = self.bn(conv)
        if self.activation:
            out = self.relu(bn)
            out = self.dropout(out)
        else:
            out = bn
        return out


class StemBlock(nn.Module):
    def __init__(self, out_ch):
        super(StemBlock, self).__init__()
        self.conv_0_1 = ConvBlock(3, out_ch, 3, 2, padding=1)
        self.conv_1_1 = ConvBlock(out_ch, out_ch // 2, 1, 1, padding=0)
        self.conv_1_2 = ConvBlock(out_ch // 2, out_ch, 3, 2, padding=1)
        self.mpool_2_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_0_2 = ConvBlock(out_ch * 2, out_ch, 3, 1, padding=1)

    def forward(self, input_tensor):
        share = self.conv_0_1(input_tensor)
        conv_branch = self.conv_1_1(share)
        conv_branch = self.conv_1_2(conv_branch)
        mpool_branch = self.mpool_2_0(share)
        out = torch.cat([conv_branch, mpool_branch], dim=1)
        out = self.conv_0_2(out)
        return out

class GatherExpansion(nn.Module):
    def __init__(self, in_ch, out_ch, strides):
        super(GatherExpansion, self).__init__()
        self.strides = strides
        if self.strides == 1:
            self.stride1_conv_1 = ConvBlock(in_ch, out_ch, 3, 1, padding=1)
            self.stride1_dwconv_1 = DWBlock(out_ch, 3, strides=1, d_multiplier=6, padding=1)
            self.stride1_conv_2 = ConvBlock(out_ch*6, out_ch, 1, 1, padding=0, activation=False)
        
        if self.strides == 2:
            self.stride2_main_dw = DWBlock(in_ch, 3, strides=2, d_multiplier=1, padding=1)
            self.stride2_main_conv = ConvBlock(in_ch, out_ch, 1, 1, padding=0, activation=False)
            self.stride2_sub_conv_1 = ConvBlock(in_ch, in_ch, 3, 1, padding=1)
            self.stride2_sub_dw_1 = DWBlock(in_ch, 3, strides=2, d_multiplier=6, padding=1)
            self.stride2_sub_dw_2 = DWBlock(in_ch*6, 3, strides=1, d_multiplier=1, padding=1)
            self.stride2_sub_conv_2 = ConvBlock(in_ch*6, out_ch, 3, 1, padding=1, activation=False)

    def forward(self, input_tensor):
        if self.strides == 1:
            branch = self.stride1_conv_1(input_tensor)
            branch = self.stride1_dwconv_1(branch)
            branch = self.stride1_conv_2(branch)
            out = branch + input_tensor
            out = F.relu(out)

        if self.strides == 2:
            branch = self.stride2_main_dw(input_tensor)
            branch = self.stride2_main_conv(branch)
            main = self.stride2_sub_conv_1(input_tensor)
            main = self.stride2_sub_dw_1(main)
            # print(main.shape)
            main = self.stride2_sub_dw_2(main)
            # print(main.shape)
            main = self.stride2_sub_conv_2(main)
            # print(main.shape)
            out = main + branch
            out = F.relu(out)
        
        return out

class DWBlock(nn.Module):
    def __init__(self, in_ch, k_size, strides, d_multiplier, padding=1):
        super(DWBlock, self).__init__()
        self.dw_conv = nn.Conv2d(in_ch, in_ch * d_multiplier, kernel_size=k_size, stride=strides, padding=padding, groups=in_ch)
        self.bn = nn.BatchNorm2d(in_ch * d_multiplier)

    def forward(self, input_tensor):
        # print(self.dw_conv, input_tensor.shape)
        out = self.dw_conv(input_tensor)
        # print(out.shape)
        out = self.bn(out)
        return out

class ContextEmbedding(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ContextEmbedding, self).__init__()
        self.ga_pool = nn.AdaptiveAvgPool2d(1)
        self.ga_pool_bn = nn.BatchNorm2d(in_ch)
        self.conv_1 = ConvBlock(in_ch, out_ch, 1, strides=1, padding=0)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, input_tensor):
        out = self.ga_pool(input_tensor)
        out = self.ga_pool_bn(out)
        out = self.conv_1(out)
        out = out + input_tensor
        out = self.conv_2(out)
        return out

class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.arch = {
            'stage_1': [[3, 64, 2, 1], [3, 64, 1, 1]],
            'stage_2': [[3, 64, 2, 1], [3, 64, 1, 2]],
            'stage_3': [[3, 128, 2, 1], [3, 128, 1, 2]]
        }
        self.layer = nn.ModuleDict()
        stage = sorted(self.arch)
        in_ch = 3
        for stage_idx in stage:
            for idx, info in enumerate(self.arch[stage_idx]):
                var = info
                k_size = var[0]
                out_ch = var[1]
                strides = var[2]
                repeat = info[3]
                for r in range(repeat):
                    # in_ch = 3 if stage_idx == 'stage_1' and idx == 0 and r == 0 else out_ch
                    self.layer[f'{stage_idx}_{idx}_{r}_conv'] = ConvBlock(in_ch, out_ch, k_size, strides, padding=1)
                    in_ch = out_ch

    def forward(self, input_tensor):
        out = input_tensor
        layer = sorted(self.layer)
        # print(layer)
        for item in layer:
            # print(item, out.shape)
            out = self.layer[item](out)
            # print(out.shape)
        return out

class SemanticBranch(nn.Module):
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.arch = {
            'stage_1': [['stem', 3, 16, 0, 4, 1]],
            'stage_3': [['ge', 3, 32, 6, 2, 1], ['ge', 3, 32, 6, 1, 1]],
            'stage_4': [['ge', 3, 64, 6, 2, 1], ['ge', 3, 64, 6, 1, 1]],
            'stage_5': [['ge', 3, 128, 6, 2, 1], ['ge', 3, 128, 6, 1, 3], ['ce', 3, 128, 0, 1, 1]]
        }
        self.layer = nn.ModuleDict()
        stage = sorted(self.arch)
        temp = None
        for stage_idx in stage:
            for idx, info in enumerate(self.arch[stage_idx]):
                var = info
                opr_type = var[0]
                k_size = var[1]
                out_ch = var[2]
                depth_multi = var[3]
                strides = var[4]
                repeat = info[5]
                for r in range(repeat):
                    if opr_type == 'stem':
                        self.layer[f'{stage_idx}_{idx}_{opr_type}_{r}'] = StemBlock(out_ch)
                    if opr_type == 'ge':
                        self.layer[f'{stage_idx}_{idx}_{opr_type}_{r}'] = GatherExpansion(temp, out_ch, strides)
                    if opr_type == 'ce':
                        self.layer[f'{stage_idx}_{idx}_{opr_type}_{r}'] = ContextEmbedding(temp, out_ch)
                    # print(temp, out_ch)
                    temp = out_ch
                    

    def forward(self, input_tensors):
        out = input_tensors
        layer = sorted(self.layer)
        # print(layer)
        # print(out.shape)
        for item in layer:
            # print(self.layer[item])
            out = self.layer[item](out)
            # print(item, out.shape)
        return out

class AggregationBranch(nn.Module):
    def __init__(self, out_ch):
        super(AggregationBranch, self).__init__()
        self.d_branch_1_dw = DWBlock(out_ch, 3, strides=1, d_multiplier=1, padding=1)
        self.d_branch_1_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.d_branch_2_conv = ConvBlock(out_ch, out_ch, 3, strides=2, padding=1, activation=False)
        self.d_branch_2_apool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.s_branch_1_dw = DWBlock(out_ch, 3, strides=1, d_multiplier=1, padding=1)
        self.s_branch_1_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.s_branch_2_conv = ConvBlock(out_ch, out_ch, 3, strides=1, padding=1, activation=False)
        self.s_branch_2_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.s_branch_3_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.main_conv = ConvBlock(out_ch, out_ch, 3, strides=1, padding=1, activation=False)

    def forward(self, detail_branch, semantic_branch):
        d_branch_main = self.d_branch_1_dw(detail_branch)
        d_branch_main = self.d_branch_1_conv(d_branch_main)
        d_branch_sub = self.d_branch_2_conv(detail_branch)
        d_branch_sub = self.d_branch_2_apool(d_branch_sub)
        s_branch_main = self.s_branch_1_dw(semantic_branch)
        s_branch_main = self.s_branch_1_conv(s_branch_main)
        s_branch_sub = self.s_branch_2_conv(semantic_branch)
        s_branch_sub = self.s_branch_2_upsample(s_branch_sub)
        s_branch_sub = torch.sigmoid(s_branch_sub)
        
        d_branch = d_branch_main * s_branch_sub
        s_branch = s_branch_main * d_branch_sub
        s_branch = self.s_branch_3_upsample(s_branch)

        out = d_branch + s_branch
        out = self.main_conv(out)
        
        return out

class BinarySegmentation(nn.Module):
    def __init__(self, cls_num):
        super(BinarySegmentation, self).__init__()
        self.conv1 = ConvBlock(128, 128, 1, strides=1, padding=0)
        self.conv2 = ConvBlock(128, 64, 1, strides=1, padding=0)
        self.conv3 = nn.Conv2d(64, cls_num, kernel_size=1, stride=1, padding=0)
        
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.upsample(out)
        out = self.softmax(out)
        
        return out

class InstanceSegmentation(nn.Module):
    def __init__(self, inst_n):
        super(InstanceSegmentation, self).__init__()
        self.conv1 = ConvBlock(128, 128, 1, strides=1, padding=0)
        self.conv2 = ConvBlock(128, 64, 1, strides=1, padding=0)
        self.conv3 = nn.Conv2d(64, inst_n, kernel_size=1, stride=1, padding=0)
        
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.upsample(out)
        
        return out

class BiseNetV2(nn.Module):
    def __init__(self):
        super(BiseNetV2, self).__init__()
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.aggregation_branch = AggregationBranch(128)
        self.bin_segmentation = BinarySegmentation(2)
        self.inst_segmentation = InstanceSegmentation(4)
    
    def forward(self, input_tensor):
        d_branch = self.detail_branch(input_tensor)
        s_branch = self.semantic_branch(input_tensor)
        agg_brang = self.aggregation_branch(d_branch, s_branch)
        
        bin_pred = self.bin_segmentation(agg_brang)
        inst_seg = self.inst_segmentation(agg_brang)
        
        return [bin_pred, inst_seg]

model = BiseNetV2()
input_tensor = torch.randn(8, 3, 512, 256)
output = model(input_tensor)
print(output[0].shape, output[1].shape)  # Should print torch.Size([1, 2, 512, 256]) torch.Size([1, 3, 512, 256])