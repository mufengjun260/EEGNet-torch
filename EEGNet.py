import torch
from torch import nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, (1, 63), padding=(0, 31))
        self.bn1_1 = torch.nn.BatchNorm2d(8)
        self.depthConv1 = torch.nn.Conv2d(8, 8 * 2, (12, 1), groups=8, bias=False)
        self.bn1_2 = torch.nn.BatchNorm2d(16)
        self.AvgPool1 = torch.nn.AvgPool2d(1, 4)

        self.conv2_1 = torch.nn.Conv2d(16, 16, (1, 15), groups=16, padding=(0, 7))
        self.conv2_2 = torch.nn.Conv2d(16, 16, 1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.AvgPool2 = torch.nn.AvgPool2d(1, 8)

        self.l1 = torch.nn.Linear(160, 4)

        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveAvgPool1d(1)
        # self.fc1_avg = torch.nn.Conv1d(64, 64, 1, bias=False)
        # self.fc2_avg = torch.nn.Conv1d(64, 64, 1, bias=False)

    def forward(self, x):
        bs, channel, length, _ = x.shape
        # ori_x = x.transpose(2, 3)

        ori_x = x.reshape(bs, 1, channel, length).type(torch.float)

        ori_x = self.conv1(ori_x)
        ori_x = self.bn1_1(ori_x)
        ori_x = self.depthConv1(ori_x)
        ori_x = self.bn1_2(ori_x)
        ori_x = F.elu(ori_x)
        ori_x = self.AvgPool1(ori_x)

        ori_x = self.conv2_2(self.conv2_1(ori_x))
        ori_x = self.bn2(ori_x)
        ori_x = F.elu(ori_x)
        ori_x = self.AvgPool2(ori_x)

        ori_x = ori_x.reshape(bs, -1)

        # channel attention
        # avg_out = self.fc2_avg(F.relu(self.fc1_avg(self.avg_pool(ori_x))))
        # max_out = self.fc2_avg(F.relu(self.fc1_avg(self.max_pool(ori_x))))
        # out = avg_out + max_out
        #
        # ori_x = ori_x * out

        ori_x = F.softmax(self.l1(ori_x), dim=1)

        return ori_x

    pass
