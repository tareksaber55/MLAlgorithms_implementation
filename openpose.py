import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class DenseConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DenseConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        return torch.cat([out1, out2, out3], dim=1)


class StageBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, internal_channels: int):
        super().__init__()
        self.blocks = nn.Sequential(
            DenseConvBlock(in_channels, internal_channels),
            DenseConvBlock(internal_channels * 3, internal_channels),
            DenseConvBlock(internal_channels * 3, internal_channels),
            DenseConvBlock(internal_channels * 3, internal_channels),
            DenseConvBlock(internal_channels * 3, internal_channels)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(internal_channels * 3, 512, kernel_size=1),
            nn.PReLU()
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=1),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.conv1x1(x)
        x = self.conv1x1_2(x)
        return x


class VGGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:23])
        self.cpm_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.cpm_conv(self.features(x))


class OpenPose(nn.Module):
    def __init__(self, num_paf_stages=4, num_cpm_stages=2, num_parts=19, num_pafs=38, internal_channels=96):
        super().__init__() 
        
        self.backbone = VGGBackbone()
        backbone_out = 128
        
        self.paf_stages = nn.ModuleList()
        self.paf_stages.append(StageBlock(backbone_out, num_pafs, internal_channels))
        
        for _ in range(1, num_paf_stages):
            self.paf_stages.append(StageBlock(num_pafs + backbone_out, num_pafs, internal_channels))
            
        self.cpm_stages = nn.ModuleList()
        self.cpm_stages.append(StageBlock(backbone_out + num_pafs, num_parts, internal_channels))
        
        for _ in range(1, num_cpm_stages):
            self.cpm_stages.append(StageBlock(num_pafs + backbone_out + num_parts, num_parts, internal_channels))

    def forward(self, x):
        F = self.backbone(x)
        
        paf_outs = []
        curr_paf_out = self.paf_stages[0](F)
        paf_outs.append(curr_paf_out)
        
        for i in range(1, len(self.paf_stages)):
            concat_features_paf = torch.cat([F, curr_paf_out], dim=1)
            curr_paf_out = self.paf_stages[i](concat_features_paf)
            paf_outs.append(curr_paf_out)

        cpm_outs = []
        curr_cpm_out = self.cpm_stages[0](torch.cat([F, curr_paf_out], dim=1))
        cpm_outs.append(curr_cpm_out)
        
        for i in range(1, len(self.cpm_stages)):
            concat_features_cpm = torch.cat([F, curr_paf_out, curr_cpm_out], dim=1)
            curr_cpm_out = self.cpm_stages[i](concat_features_cpm)
            cpm_outs.append(curr_cpm_out)

        return paf_outs, cpm_outs