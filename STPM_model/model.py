import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import conv1x1, BasicBlock
from torchvision.models.resnet import ResNet18_Weights


class MResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=1,
                 width_per_group=64, out_features = ["1","2","3","4"], downscale=False):
        super(MResNet18, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.downscale = downscale
        self.out_features = out_features


        # modules initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


# the output activations from each of the 3 blocks are returned at each forward-pass
# (these are the activations that will be compared to compute the anomaly maps)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xs = {"1": x1, "2": x2, "3": x3, "4": x4}

        return [xs[f] for f in self.out_features]
    




# function to easily instantiate the models

def modified_resnet18(pretrained, **kwargs):
    model = MResNet18(block = BasicBlock, layers = [2,2,2,2], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(ResNet18_Weights.IMAGENET1K_V1.url, progress = True)
        model.load_state_dict(state_dict, strict=False)
    return model
     