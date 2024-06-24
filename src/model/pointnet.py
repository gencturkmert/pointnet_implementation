import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, channels):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.layers.append(nn.Conv1d(channels[i], channels[i+1], 1))
            if i < len(channels) - 2:  # No batch norm and activation in the last layer
                self.layers.append(nn.BatchNorm1d(channels[i+1]))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.mlp1 = MLP([k, 64, 128, 1024])
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
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.fc_bn(self.fc(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)

        init_matrix = self.identity.repeat(batch_size, 1)
        x = x + init_matrix
        x = x.view(-1, self.k, self.k)
        return x
    
class InputTransform(nn.Module):
    def __init__(self):
        super(InputTransform, self).__init__()
        self.transform = TNet(k=3)  # 3D points

    def forward(self, x):
        input_transform = self.transform(x)
        x = torch.bmm(x.transpose(1, 2), input_transform).transpose(1, 2)
        return x, input_transform

class FeatureTransform(nn.Module):
    def __init__(self):
        super(FeatureTransform, self).__init__()
        self.transform = TNet(k=64)  # Feature dimension

    def forward(self, x):
        feature_transform = self.transform(x)
        x = torch.bmm(x.transpose(1, 2), feature_transform).transpose(1, 2)
        return x, feature_transform

class GlobalFeatureAggregation(nn.Module):
    def __init__(self):
        super(GlobalFeatureAggregation, self).__init__()

    def forward(self, x):
        # Max pooling over the points dimension (dim=2)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(x.size(0), -1)  # Flatten the features for each sample in the batch
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
        x = self.drop1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)  # No activation here, use CrossEntropyLoss which includes LogSoftmax
        return x
    

class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.input_transform = InputTransform()
        self.feature_transform = FeatureTransform()
        self.mlp1 = MLP([3, 64, 64])
        self.mlp2 = MLP([64, 128, 1024])
        self.global_features = GlobalFeatureAggregation()
        self.class_head = ClassificationHead(num_classes)

    def forward(self, x):
        x, input_trans = self.input_transform(x)
        x = self.mlp1(x)
        x, feature_trans = self.feature_transform(x)
        x = self.mlp2(x)
        x = self.global_features(x)
        x = self.class_head(x)
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
        global_features = global_features.view(-1, 1024, 1).repeat(1, 1, self.num_points)  # Repeat global features for each point
        x = torch.cat((x, global_features), 1)  # Concatenate global features with point features
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # No activation, output size: (batch_size, num_classes, num_points)
        return x
    
class PointNetSegmentation(nn.Module):
    def __init__(self, num_points, num_classes):
        super(PointNetSegmentation, self).__init__()
        self.num_points = num_points
        self.input_transform = InputTransform()
        self.mlp1 = MLP([3, 64, 64])
        self.feature_transform = FeatureTransform()
        self.mlp2 = MLP([64, 64, 128, 1024])
        self.global_features = GlobalFeatureAggregation()
        self.segmentation_head = SegmentationHead(num_points, num_classes)

    def forward(self, x):
        x, input_trans = self.input_transform(x)
        x = self.mlp1(x)
        x, feature_trans = self.feature_transform(x)
        point_features = self.mlp2(x)
        global_features = self.global_features(point_features)
        x = self.segmentation_head(point_features, global_features)
        return x.transpose(2, 1), input_trans, feature_trans  # Transpose to match the shape (batch_size, num_points, num_classes)



