import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchsummaryX import summary
from torch.hub import load_state_dict_from_url
from .configuration import voc_ssd300_configuration as config


class SSD(nn.Module):
    def __init__(self, mode):
        self.mode = mode
        super().__init__()
        # vgg->conv1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # vgg->conv2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # vgg->conv3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # vgg->conv4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # num_anchors * (num_classes+background)
        self.conf4_3 = nn.Conv2d(
            512,
            config["layer_num_anchors"][0] * config["num_classes"],
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.loc4_3 = nn.Conv2d(
            512,
            config["layer_num_anchors"][0] * 4,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        # vgg->conv5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # extra->conv6
        self.conv6_1 = nn.Conv2d(
            512, 1024, kernel_size=3, stride=1, padding=6, dilation=6
        )
        self.bn6_1 = nn.BatchNorm2d(1024)

        # extra->conv7
        self.conv7_1 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.bn7_1 = nn.BatchNorm2d(1024)
        self.conf7_1 = nn.Conv2d(
            1024,
            config["layer_num_anchors"][1] * config["num_classes"],
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.loc7_1 = nn.Conv2d(
            1024,
            config["layer_num_anchors"][1] * 4,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        # extra->conv8
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.bn8_1 = nn.BatchNorm2d(256)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn8_2 = nn.BatchNorm2d(512)
        self.conf8_2 = nn.Conv2d(
            512,
            config["layer_num_anchors"][2] * config["num_classes"],
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.loc8_2 = nn.Conv2d(
            512,
            config["layer_num_anchors"][2] * 4,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        # extra->conv9
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.bn9_1 = nn.BatchNorm2d(128)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn9_2 = nn.BatchNorm2d(256)
        self.conf9_2 = nn.Conv2d(
            256,
            config["layer_num_anchors"][3] * config["num_classes"],
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.loc9_2 = nn.Conv2d(
            256,
            config["layer_num_anchors"][3] * 4,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        # extra->conv10
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn10_1 = nn.BatchNorm2d(128)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn10_2 = nn.BatchNorm2d(256)
        self.conf10_2 = nn.Conv2d(
            256,
            config["layer_num_anchors"][4] * config["num_classes"],
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.loc10_2 = nn.Conv2d(
            256,
            config["layer_num_anchors"][4] * 4,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        # extra->conv11
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conf11_2 = nn.Conv2d(
            256,
            config["layer_num_anchors"][5] * config["num_classes"],
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.loc11_2 = nn.Conv2d(
            256,
            config["layer_num_anchors"][5] * 4,
            kernel_size=3,
            padding=1,
            stride=1,
        )

    def forward(self, x):
        confs, locs = [], []

        # conv1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        # conv2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        # conv3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)

        # conv4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.bn4_3(self.conv4_3(x))
        confs.append(self.conf4_3(x))
        locs.append(self.loc4_3(x))
        x = self.pool4(F.relu(x))

        # conv5
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = self.pool5(x)

        # conv6
        x = F.relu(self.bn6_1(self.conv6_1(x)))

        # conv7
        x = self.bn7_1(self.conv7_1(x))
        confs.append(self.conf7_1(x))
        locs.append(self.loc7_1(x))
        x = F.relu(x)

        # conv8
        x = F.relu(self.bn8_1(self.conv8_1(x)))
        x = self.bn8_2(self.conv8_2(x))
        confs.append(self.conf8_2(x))
        locs.append(self.loc8_2(x))
        x = F.relu(x)

        # conv9
        x = F.relu(self.bn9_1(self.conv9_1(x)))
        x = self.bn9_2(self.conv9_2(x))
        confs.append(self.conf9_2(x))
        locs.append(self.loc9_2(x))
        x = F.relu(x)

        # conv10
        x = F.relu(self.bn10_1(self.conv10_1(x)))
        x = self.bn10_2(self.conv10_2(x))
        confs.append(self.conf10_2(x))
        locs.append(self.loc10_2(x))
        x = F.relu(x)

        # conv11
        x = F.relu(self.conv11_1(x))
        x = self.conv11_2(x)
        confs.append(self.conf11_2(x))
        locs.append(self.loc11_2(x))

        confs = [conf.permute(0, 2, 3, 1).contiguous() for conf in confs]
        confs = torch.cat([conf.view(conf.size(0), -1) for conf in confs], dim=1)
        locs = [loc.permute(0, 2, 3, 1).contiguous() for loc in locs]
        locs = torch.cat([loc.view(loc.size(0), -1) for loc in locs], dim=1)

        if self.mode == "train":
            return (
                confs.view(confs.size(0), -1, config["num_classes"]),
                locs.view(locs.size(0), -1, 4),
            )
        else:
            return (
                F.softmax(
                    confs.view(confs.size(0), -1, config["num_classes"]),
                    dim=-1,
                ),
                locs.view(locs.size(0), -1, 4),
            )

    def loadPretrainedWeights(self):
        vgg_state_dict = load_state_dict_from_url(config["state_dict_url"])
        pretrained_model_keys = list(vgg_state_dict.keys())
        pretrained_model_values = list(vgg_state_dict.values())
        ssd_model_keys = list(self.state_dict().keys())
        ssd_model_values = list(self.state_dict().values())

        new_state_dict = {}
        for i in range(len(pretrained_model_keys)):
            if np.shape(pretrained_model_values[i]) == np.shape(ssd_model_values[i]):
                new_state_dict[ssd_model_keys[i]] = pretrained_model_values[i]
        self.load_state_dict(new_state_dict, strict=False)

    def initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def loadTrainedWeights(self, weights_file_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(weights_file_path, map_location=device))


def showNetworkMainStructure():
    model = SSD("train")
    summary(model, torch.randn(1, 3, config["image_size"], config["image_size"]))


def showOutputTensorShape():
    model = SSD("train")
    x = torch.randn(1, 3, config["image_size"], config["image_size"])
    confs, locs = model(x)
    print(confs.shape, locs.shape)


if __name__ == "__main__":
    # showNetworkMainStructure()
    # showOutputTensorShape()
    pass
