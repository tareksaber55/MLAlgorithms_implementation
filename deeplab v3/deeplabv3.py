import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Atrous Spatial Pyramid Pooling (ASPP)
# ==========================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3 if dilation > 1 else 1,
                padding=dilation if dilation > 1 else 0,
                dilation=dilation,
                bias=False  # Fixed: Use False instead of 0
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(input=x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: list, out_channels: int = 256, dropout: float = 0.5):
        super(ASPP, self).__init__()
        modules = []

        # Branch 1: 1x1 Conv
        modules.append(ASPPConv(in_channels=in_channels, out_channels=out_channels, dilation=1))

        # Branch 2-4: 3x3 Dilated Convs
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels=in_channels, out_channels=out_channels, dilation=rate))

        # Branch 5: Image-level Pooling
        modules.append(ASPPPooling(in_channels=in_channels, out_channels=out_channels))

        self.convs = nn.ModuleList(modules)

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * len(self.convs), out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.projection(res)


# ==========================================
# 2. ResNet Backbone
# ==========================================
class Bottleneck(nn.Module):
    expansion = 4  # Class attribute so `block.expansion` works outside instance

    def __init__(self, in_planes: int, planes: int, stride: int = 1, dilation: int = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            in_channels=planes, out_channels=planes, kernel_size=3,
            stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.b2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * self.expansion, kernel_size=1, bias=False)
        self.b3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.b3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    def __init__(self, block: Bottleneck, layers: list = [3, 4, 23, 3]):
        super(ResNetBackbone, self).__init__()  # Fixed: Call super().__init__()
        self.inplanes = 64  # Fixed: Must be 64, not 4

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),  # Fixed: pass stride
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, dilation if stride == 1 else 1, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):  # Fixed: range(1, blocks) instead of len(blocks)-1
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)  # Fixed: self.b1 instead of self.bn1
        x = self.relu(x)
        x = self.max_pool(x)  # Fixed: self.max_pool instead of self.maxpool

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# ==========================================
# 3. DeepLabV3 Model Wrapper
# ==========================================
class DeepLabV3(nn.Module):
    def __init__(self, num_classes: int = 21, backbone: str = 'resnet101'):
        super(DeepLabV3, self).__init__()

        if backbone == 'resnet50':
            self.backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
        elif backbone == 'resnet101':
            self.backbone = ResNetBackbone(Bottleneck, [3, 4, 23, 3])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        atrous_rates = [6, 12, 18]

        self.aspp = ASPP(in_channels=2048, atrous_rates=atrous_rates, out_channels=256)

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]

        features = self.backbone(x)
        aspp_features = self.aspp(features)
        logits = self.classifier(aspp_features)
        out = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)

        return out


if __name__ == "__main__":
    print("--- Verifying DeepLabV3 Implementation ---")

    batch_size = 2
    num_classes = 21
    input_height, input_width = 512, 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeepLabV3(num_classes=num_classes, backbone='resnet50').to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, input_height, input_width, device=device)

    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (batch_size, num_classes, input_height, input_width)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"

    print(f"Device used: {device}")
    print(f"Input Tensor Shape:  {dummy_input.shape}")
    print(f"Output Tensor Shape: {output.shape}")
    print("Forward Pass: SUCCESS!")

    # Verify Backward Pass
    model.train()
    target_masks = torch.randint(0, num_classes, (batch_size, input_height, input_width), device=device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    train_output = model(dummy_input)
    loss = criterion(train_output, target_masks)
    loss.backward()
    optimizer.step()

    print(f"Training Step Loss: {loss.item():.4f}")
    print("Backward Pass: SUCCESS!")