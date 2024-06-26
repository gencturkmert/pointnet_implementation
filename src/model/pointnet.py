import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = __name__ == "__main__"

class BasicMLP(nn.Module):
    def __init__(self, channels):
        super(BasicMLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.layers.append(nn.Conv1d(channels[i], channels[i+1], 1))
            if i < len(channels) - 2:  # No batch norm and activation in the last layer
                self.layers.append(nn.BatchNorm1d(channels[i+1]))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        if DEBUG: print(f"MLP Layer Input Shape: {x.shape}")
        for layer in self.layers:
            x = layer(x)
            if DEBUG: print(f"MLP Layer Output Shape: {x.shape}")
        return x

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.mlp1 = BasicMLP([k, 64, 128, 1024])
        self.fc = nn.Linear(1024, 512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k*k)

        self.relu = nn.ReLU()

        self.identity = nn.Parameter(torch.eye(k).flatten(), requires_grad=False)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.mlp1(x)
        if DEBUG: print(f"TNet MLP1 Output Shape: {x.shape}")

        x = torch.max(x, 2, keepdim=True)[0]
        if DEBUG: print(f"TNet Max Pooling Output Shape: {x.shape}")

        x = x.view(batch_size, -1)
        if DEBUG: print(f"TNet Flattened Output Shape: {x.shape}")

        x = self.relu(self.fc_bn(self.fc(x)))
        if DEBUG: print(f"TNet FC1 Output Shape: {x.shape}")

        x = self.relu(self.fc2_bn(self.fc2(x)))
        if DEBUG: print(f"TNet FC2 Output Shape: {x.shape}")

        x = self.fc3(x)
        if DEBUG: print(f"TNet FC3 Output Shape: {x.shape}")

        init_matrix = self.identity.repeat(batch_size, 1)
        x = x + init_matrix
        x = x.view(-1, self.k, self.k)
        if DEBUG: print(f"TNet Final Output Shape: {x.shape}")
        return x

class InputTransform(nn.Module):
    def __init__(self):
        super(InputTransform, self).__init__()
        self.transform = TNet(k=3)  # 3D points

    def forward(self, x):
        batch_size, num_points, _ = x.size()
        x = x.transpose(1, 2)  # Transpose to shape [batch_size, channels, points]
        if DEBUG: print(f"InputTransform Transposed Input Shape: {x.shape}")

        input_transform = self.transform(x)
        x = x.transpose(1, 2)  # Transpose back to [batch_size, points, channels]
        if DEBUG: print(f"InputTransform Transposed Back Shape: {x.shape}")

        x = torch.bmm(x, input_transform)  # Batch matrix multiplication
        if DEBUG: print(f"InputTransform Batch Matrix Multiplication Output Shape: {x.shape}")

        return x, input_transform

class FeatureTransform(nn.Module):
    def __init__(self):
        super(FeatureTransform, self).__init__()
        self.transform = TNet(k=64)  # Feature dimension

    def forward(self, x):
        batch_size, num_points, channels = x.size()
        if DEBUG: print(f"FeatureTransform Transposed Input Shape: {x.shape}")

        feature_transform = self.transform(x)
        x = x.transpose(1, 2)  # Transpose back to [batch_size, points, channels]
        if DEBUG: print(f"FeatureTransform Transposed Back Shape: {x.shape}")

        x = torch.bmm(x, feature_transform)  # Batch matrix multiplication
        if DEBUG: print(f"FeatureTransform Batch Matrix Multiplication Output Shape: {x.shape}")

        return x, feature_transform

class GlobalFeatureAggregation(nn.Module):
    def __init__(self):
        super(GlobalFeatureAggregation, self).__init__()

    def forward(self, x):
        # Max pooling over the points dimension (dim=2)
        x = torch.max(x, 2, keepdim=True)[0]
        if DEBUG: print(f"GlobalFeatureAggregation Max Pooling Output Shape: {x.shape}")

        x = x.view(x.size(0), -1)  # Flatten the features for each sample in the batch
        if DEBUG: print(f"GlobalFeatureAggregation Flattened Output Shape: {x.shape}")

        return x

class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        if DEBUG: print(f"ClassificationHead FC1 Output Shape: {x.shape}")

        x = self.drop1(x)
        if DEBUG: print(f"ClassificationHead Dropout1 Output Shape: {x.shape}")

        x = self.relu(self.bn2(self.fc2(x)))
        if DEBUG: print(f"ClassificationHead FC2 Output Shape: {x.shape}")

        x = self.drop2(x)
        if DEBUG: print(f"ClassificationHead Dropout2 Output Shape: {x.shape}")

        x = self.fc3(x)  # No activation here, use CrossEntropyLoss which includes LogSoftmax
        if DEBUG: print(f"ClassificationHead FC3 Output Shape: {x.shape}")

        return x

class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.input_transform = InputTransform()
        self.feature_transform = FeatureTransform()
        self.mlp1 = BasicMLP([3, 64, 64])
        self.mlp2 = BasicMLP([64, 128, 1024])
        self.global_features = GlobalFeatureAggregation()
        self.class_head = ClassificationHead(num_classes)

    def forward(self, x):
        if DEBUG: print(f"PointNet Input Shape: {x.shape}")

        x, input_trans = self.input_transform(x)
        if DEBUG: print(f"PointNet InputTransform Output Shape: {x.shape}")

        x = x.transpose(2, 1)  # Transpose to shape [batch_size, channels, points]
        if DEBUG: print(f"PointNet MLP1 Input Shape: {x.shape}")
        x = self.mlp1(x)
        if DEBUG: print(f"PointNet MLP1 Output Shape: {x.shape}")

        x, feature_trans = self.feature_transform(x)
        if DEBUG: print(f"PointNet FeatureTransform Output Shape: {x.shape}")

        x = x.transpose(2, 1)  # Transpose to shape [batch_size, channels, points]
        x = self.mlp2(x)
        if DEBUG: print(f"PointNet MLP2 Output Shape: {x.shape}")

        x = self.global_features(x)
        if DEBUG: print(f"PointNet GlobalFeatureAggregation Output Shape: {x.shape}")

        x = self.class_head(x)
        if DEBUG: print(f"PointNet ClassificationHead Output Shape: {x.shape}")

        return x, input_trans, feature_trans

class SegmentationHead(nn.Module):
    def __init__(self, num_points, num_classes):
        super(SegmentationHead, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(1088, 512, 1)  # 1024 global features + 64 local features
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.relu = nn.ReLU()

    def forward(self, x, global_features):
        if DEBUG: 
            print(f"SegmentationHead Input Shape: {x.shape}")
            print(f"SegmentationHead Global Features Shape: {global_features.shape}")

        global_features = global_features.view(-1, 1024, 1).repeat(1, 1, self.num_points)  # Repeat global features for each point
        if DEBUG: print(f"SegmentationHead Repeated Global Features Shape: {global_features.shape}")

        x = torch.cat((x, global_features), 1)  # Concatenate global features with point features
        if DEBUG: print(f"SegmentationHead Concatenated Shape: {x.shape}")

        x = self.relu(self.bn1(self.conv1(x)))
        if DEBUG: print(f"SegmentationHead Conv1 Output Shape: {x.shape}")

        x = self.relu(self.bn2(self.conv2(x)))
        if DEBUG: print(f"SegmentationHead Conv2 Output Shape: {x.shape}")

        x = self.relu(self.bn3(self.conv3(x)))
        if DEBUG: print(f"SegmentationHead Conv3 Output Shape: {x.shape}")

        x = self.conv4(x)  # No activation, output size: (batch_size, num_classes, num_points)
        if DEBUG: print(f"SegmentationHead Conv4 Output Shape: {x.shape}")

        return x

class PointNetSegmentation(nn.Module):
    def __init__(self, num_points, num_classes):
        super(PointNetSegmentation, self).__init__()
        self.num_points = num_points
        self.input_transform = InputTransform()
        self.mlp1 = BasicMLP([3, 64, 64])
        self.feature_transform = FeatureTransform()
        self.mlp2 = BasicMLP([64, 64, 128, 1024])
        self.global_features = GlobalFeatureAggregation()
        self.segmentation_head = SegmentationHead(num_points, num_classes)

    def forward(self, x):
        if DEBUG: print(f"PointNetSegmentation Input Shape: {x.shape}")

        x, input_trans = self.input_transform(x)
        if DEBUG: print(f"PointNetSegmentation InputTransform Output Shape: {x.shape}")

        x = x.transpose(2, 1)  # Transpose to shape [batch_size, channels, points]
        x = self.mlp1(x)
        if DEBUG: print(f"PointNetSegmentation MLP1 Output Shape: {x.shape}")

        x, feature_trans = self.feature_transform(x)
        if DEBUG: print(f"PointNetSegmentation FeatureTransform Output Shape: {x.shape}")

        x = x.transpose(2, 1)  # Transpose to shape [batch_size, channels, points]
        x = self.mlp2(x)
        if DEBUG: print(f"PointNetSegmentation MLP2 Output Shape: {x.shape}")

        global_features = self.global_features(x)
        if DEBUG: print(f"PointNetSegmentation GlobalFeatureAggregation Output Shape: {global_features.shape}")

        x = self.segmentation_head(x, global_features)
        if DEBUG: print(f"PointNetSegmentation SegmentationHead Output Shape: {x.shape}")

        return x.transpose(2, 1), input_trans, feature_trans  # Transpose to match the shape (batch_size, num_points, num_classes)

# Example usage for debugging:
if __name__ == "__main__":
    num_points = 1024
    num_classes = 40
    batch_size = 32
    dummy_input = torch.rand(batch_size, num_points, 3)

    model = PointNetSegmentation(num_points, num_classes)
    outputs, input_trans, feature_trans = model(dummy_input)
    print(f"Final Output Shape: {outputs.shape}")
    print(f"Input Transformation Matrix Shape: {input_trans.shape}")
    print(f"Feature Transformation Matrix Shape: {feature_trans.shape}")

    model_seg = PointNet(num_classes)
    outputs_seg, input_trans_seg, feature_trans_seg = model_seg(dummy_input)
    print(f"Class Final Output Shape: {outputs_seg.shape}")
    print(f"Class Input Transformation Matrix Shape: {input_trans_seg.shape}")
    print(f"Class Feature Transformation Matrix Shape: {feature_trans_seg.shape}")
