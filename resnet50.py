import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

'''
卷积：
步长决定滑动窗口步子大小,也就决定了特征图缩放比例
首先在满足2 * padding = kernel_size - 1的情况下
步长等于1，特征图尺寸不变；步长等于2，特征图尺寸减半

池化
通常情况下池化层不使用填充，池化的目的是下采样来减少特征图的尺寸和计算量
特征图的尺寸依然和步长挂钩
'''

# 先构建残差块，再去构建残差层，残差层是残差块的堆叠
# resnet18只用了基础残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, block_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channel, block_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(block_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(block_channel, block_channel * self.expansion, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(block_channel)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu1(self.bn1(self.conv1(x)))  # 卷积、批归一化、relu
        out = self.bn2(self.conv2(out))  # 经过两层后的结果和原始输入进行相加，实现残差块的残差连接，
        out += identity
        out = self.relu2(out)  # 可以注意到第二层卷积bn后先加了输入，再去relu
        return out


# resnet50只用了这种残差块
'''
残差块只有第一个卷积层的步长用形参stride来传，默认传2，其余地方写固定步长1。
实际上只有在每个layer的第一个卷积层里需要降尺寸。
'''
class bottleneck(nn.Module):
    expansion = 4  # 扩展通道数至：通道数*expansion

    def __init__(self, in_channel, block_channel, stride=1, downsample=None):
        super(bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channel, block_channel, kernel_size=1, stride=stride, bias=False)  # 这里padding = 0
        self.bn1 = nn.BatchNorm2d(block_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(block_channel, block_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(block_channel)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(block_channel, block_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(block_channel * self.expansion)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu3(out)
        return out


class Resnet(nn.Module):
    def __init__(self, in_channel=3, num_classes=100, block=bottleneck, num_blocks=[3, 4, 6, 3]):
        super(Resnet, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channel = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3],stride=2)
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion*7*7, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.maxpool1(self.bn1(self.conv1(x)))  # (1, 3, 224, 224) -> (1, 64, 56, 56)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.shape[0], -1)  # out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def _make_layer(self, block, block_channel, block_num, stride):
        layers = []
        # 注意：第二个layer的输入是256通道个56*56，下采样后是512通道个28*28。
        downsample = nn.Conv2d(self.in_channel, block_channel * block.expansion, kernel_size=1, stride=stride,
                               bias=False)

        # 先加一个带有下采样的layer
        layers += [block(self.in_channel, block_channel, stride=stride, downsample=downsample)]
        self.in_channel = block_channel * block.expansion

        # 再加block_num-1个默认不带下采样的layer，由于输入输出通道数相同，所以不需要下采样
        for _ in range(1, block_num):
            layers += [block(self.in_channel, block_channel, stride=1)]
        return nn.Sequential(*layers)


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    my_resnet50 = Resnet()
    resnet50 = torchvision.models.resnet50()  # 看源码对比官方的resnet50

    print(my_resnet50)
    # print(resnet50)

    y = my_resnet50(x)
    print(y.shape)
