import torch
import torch.nn as nn
import torchvision.models as models

class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        self.num_classes = num_classes
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = resnet.fc.in_features
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        self.classifier = nn.Linear(in_features, num_classes)
        self.regressor = nn.Linear(in_features, 4 * num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)                # (B, C, H, W)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)          # (B, C, 1, 1)
        x = torch.flatten(x, 1)                      # (B, C)
        
        classes = self.classifier(x)                 # (B, num_classes)
        coordinates = self.regressor(x)              # (B, 4*num_classes)
        coordinates = coordinates.view(-1, self.num_classes, 4)
        
        return classes, coordinates
