import torch
import torch.nn as nn
from torchsummary import summary
from collections import defaultdict
import torch.nn.functional as F


###############################################特殊结构##################################################################


def double_conv(in_channels, out_channels):
    """
    description: U-Net中的block结构，包含两个基础卷积和一个maxpool但是由于pool前的数据需要跨层传输所以这边不包含maxpool结构
    in_channels: 传入数据的通道数量
    out_channels: 输出数据的通道数量
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


################################################神经网络模型##############################################################


class Unet(nn.Module):
    """
    description: Unet model 结构
    n_class：分类的类别
    """

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        out = self.sigmoid(x)

        return out


######################################################损失函数###########################################################


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def my_test_loss(pred, target, metrics, bce_weight=0.5):
    """
    description: 损失函数，有两个部分组成，分别是dice_loss和binary_cross_entropy_loss
    pred: 神经网络的预测结果
    target: 真实标签
    metrics: 记录结果的字典
    bce_weight: 两种损失函数的占比，binary_cross_entropy_loss的比重为bce_weight,dice_loss的占比则为1-bce_weight
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)

    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


if __name__ == '__main__':
    x = torch.ones([1, 1, 512, 512])
    unet = Unet(1)
    summary(unet, input_size=(1, 512, 512))
    out = unet.forward(x)
    metrics = defaultdict(float)
    loss = my_test_loss(out,x,metrics)
    print(loss)
