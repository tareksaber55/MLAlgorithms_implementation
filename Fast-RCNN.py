import torch
import torch.nn as nn
from torchvision.ops import roi_pool # you can simply use this
import torchvision.models as models
import torch.nn.functional as F
class ROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.pool = nn.MaxPool2d(output_size)

    def forward(self, feature_map, rois):
        # feature_map: (B, C, H, W)
        # rois: (num_rois, 5)
        # each ROI: [batch_index, x1, y1, x2, y2]

        B, C, H, W = feature_map.shape
        rois_num = rois.shape[0]

        pooled_rois = torch.zeros(
            rois_num,
            C,
            self.output_size[0],
            self.output_size[1],
            device=feature_map.device,
            dtype=feature_map.dtype
        )

        for i in range(rois_num):

            roi = rois[i]

            batch_index = int(roi[0])

            x1 = int(torch.round(roi[1] * self.spatial_scale).item())
            y1 = int(torch.round(roi[2] * self.spatial_scale).item())
            x2 = int(torch.round(roi[3] * self.spatial_scale).item())
            y2 = int(torch.round(roi[4] * self.spatial_scale).item())

            # Clamp coordinates to feature-map boundaries
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))

            # Check batch index
            if not 0 <= batch_index < B:
                raise ValueError(
                    f"Invalid batch index {batch_index} for batch size {B}"
                )

            # Check ROI validity
            if x2 < x1 or y2 < y1:
                raise ValueError(
                    f"Invalid ROI: {[batch_index, x1, y1, x2, y2]}"
                )

            roi_feature_map = feature_map[
                batch_index,
                :,
                y1:y2 + 1,
                x1:x2 + 1
            ]

            roi_pooled = self.pool(roi_feature_map)

            pooled_rois[i] = roi_pooled

        return pooled_rois

            





class FastRCNN(nn.Module):
    def __init__(self,num_classes:int):
        super(FastRCNN, self).__init__()
        vgg = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:-1])
        # input:       224 x 224
        # conv/pool -> 14 x 14
        # VGG16 feature map has 512 channels
        self.roi_pool = ROIPool(output_size=(7,7),spatial_scale=1/16)
        self.fc1 = nn.Linear(512*7*7,4096)
        self.fc2 = nn.Linear(4096,4096)
        # out head
        # K object classes + background
        self.classifier = nn.Linear(4096,num_classes+1)
        self.regressor = nn.Linear(4096,num_classes*4)
    def forward(self, images, rois):
        feature_maps = self.backbone(images)

        roi_features = self.roi_pool(
            feature_maps,
            rois
        )

        roi_features = torch.flatten(roi_features,start_dim=1)

        roi_features = F.relu(self.fc1(roi_features))
        roi_features = F.relu(self.fc2(roi_features))

        class_scores = self.classifier(roi_features)
        bbox_deltas = self.regressor(roi_features)

        return class_scores, bbox_deltas




        

