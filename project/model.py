
import torch
import torch.nn as nn

from pathlib import Path

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()

        # Convolutional blocks
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )

        # Global adaptive pool
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 100)
        )
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)

        return x


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_dropout = nn.Dropout(p=0.5)

        self.layer1 = self.make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self.make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self.make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self.make_layer(256, 512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 100)

        self.init_weights()

    def make_layer(self, in_ch, out_ch, blocks, stride):
        layer = nn.ModuleList()
        for i in range(blocks):
            s = stride if i == 0 else 1
            in_c = in_ch if i == 0 else out_ch
            block = nn.ModuleDict({
                "conv1": conv3x3(in_c, out_ch, s),
                "bn1": nn.BatchNorm2d(out_ch),
                "conv2": conv3x3(out_ch, out_ch, 1),
                "bn2": nn.BatchNorm2d(out_ch),
            })
            if s != 1 or in_c != out_ch:
                block["downsample"] = nn.Sequential(
                    conv1x1(in_c, out_ch, s),
                    nn.BatchNorm2d(out_ch),
                )
            layer.append(block)

        return layer

    def forward_layer(self, x, layer):
        for block in layer:
            identity = x
            out = block["conv1"](x)
            out = block["bn1"](out)
            out = self.relu(out)
            out = self.dropout(out)
            out = block["conv2"](out)
            out = block["bn2"](out)
            if "downsample" in block:
                identity = block["downsample"](x)
            x = self.relu(out + identity)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.forward_layer(x, self.layer1)
        x = self.forward_layer(x, self.layer2)
        x = self.forward_layer(x, self.layer3)
        x = self.forward_layer(x, self.layer4)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self.make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self.make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self.make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 100)

        self.init_weights()

    def make_layer(self, in_ch, out_ch, blocks, stride):
        layer = nn.ModuleList()
        for i in range(blocks):
            s = stride if i == 0 else 1
            in_c = in_ch if i == 0 else out_ch
            block = nn.ModuleDict({
                "conv1": conv3x3(in_c, out_ch, s),
                "bn1": nn.BatchNorm2d(out_ch),
                "conv2": conv3x3(out_ch, out_ch, 1),
                "bn2": nn.BatchNorm2d(out_ch),
            })
            if s != 1 or in_c != out_ch:
                block["downsample"] = nn.Sequential(
                    conv1x1(in_c, out_ch, s),
                    nn.BatchNorm2d(out_ch),
                )
            layer.append(block)

        return layer

    def forward_layer(self, x, layer):
        for block in layer:
            identity = x
            out = block["conv1"](x)
            out = block["bn1"](out)
            out = self.relu(out)
            out = block["conv2"](out)
            out = block["bn2"](out)
            if "downsample" in block:
                identity = block["downsample"](x)
            x = self.relu(out + identity)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.forward_layer(x, self.layer1)
        x = self.forward_layer(x, self.layer2)
        x = self.forward_layer(x, self.layer3)
        x = self.forward_layer(x, self.layer4)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

