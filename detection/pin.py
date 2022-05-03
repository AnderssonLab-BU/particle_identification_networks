import torch

from detection.utils import FocalLoss


class ParticleIdentificationNetwork(torch.nn.Module):
    def __init__(self, model_cfg=None):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(16, 1)
        self.loss = torch.nn.BCELoss(reduction="sum")
        # self.loss = FocalLoss(alpha=0.65, gamma=2.0, reduction="sum")

    def forward(self, img):
        img = torch.unsqueeze(img, dim=1)
        # input shape should be [batch_size, c=1, w, h]a
        c1 = self.conv1(img)
        c1 = torch.relu(c1)
        c1 = self.pooling(c1)
        c2 = self.conv2(c1)
        c2 = torch.relu(c2)
        c2 = self.pooling(c2)
        c3 = self.conv3(c2)
        c3 = torch.relu(c3)
        c3 = self.pooling(c3)
        # permute axis from [batch, channel, w, h] ([0,1,2,3])
        # to [batch, w, h, channel]([0,2,3,1])
        c3 = c3.permute([0, 2, 3, 1])
        out = self.fc1(c3)
        out = torch.squeeze(out, dim=-1)
        out = torch.sigmoid(out)
        return out

    def compute_loss(self, input, rank):
        # target heatmap, [batch_size, w, h]
        batch_size = input["gt_heatmap"].shape[0]
        avg_loss = self.loss(self.forward(
            input["img"].cuda(rank)), input["gt_heatmap"].cuda(rank))/batch_size
        return avg_loss
