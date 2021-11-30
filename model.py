import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

class Pre(nn.Module):
    def __init__(self, in_channel=3, out_channel=64):
        super(Pre, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class Post(nn.Module):
    def __init__(self, in_channel=64):
        super(Post, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=3, kernel_size=3, stride=1)

    def forward(self, x):
        out = self.conv(self.pad(x))
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_channel, out_channel, expansion=1):
        super(DenseLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        inter_channel = in_channel * expansion

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=inter_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=inter_channel, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv3 = nn.Conv2d(out_channel*2, in_channel, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return self.conv3(torch.cat((x, out), dim=1))

class TransLayer(nn.Module):
    def __init__(self, in_channel):
        super(TransLayer, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1)
        )
        self.relu = nn.PReLU()

    def forward(self, x):
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, res)
        return out

class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        x = x * y

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        z = torch.cat([avg_out, max_out], dim=1)
        z = self.conv(z)
        z = self.Sigmoid(z)

        return z * x

class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()

        self.pre = Pre()

        haze_class = models.densenet121(pretrained=True)

        ############# Block1-down ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        ############# Block3-down ##############
        self.dense_block4 = haze_class.features.denseblock4
        self.trans_block4 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1)),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.res = nn.Sequential(
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024)
        )

        self.dense4 = DenseLayer(2048, 2048)
        self.trans4 = TransLayer(2048)
        self.atten4 = Attention(512)

        self.dense3 = DenseLayer(1024, 1024)
        self.trans3 = TransLayer(1024)
        self.atten3 = Attention(256)

        self.dense2 = DenseLayer(512, 512)
        self.trans2 = TransLayer(512)
        self.atten2 = Attention(128)

        self.dense1 = DenseLayer(256, 256)
        self.trans1 = TransLayer(256)
        self.atten1 = Attention(64)

        self.post = Post()

    def forward(self, x):
        x1 = self.pre(x) ## 64, 256, 256
        x1 = self.trans_block1(self.dense_block1(x1)) ## 128, 128, 128
        x2 = self.trans_block2(self.dense_block2(x1)) ## 256, 64, 64
        x3 = self.trans_block3(self.dense_block3(x2)) ## 512, 32, 32
        x4 = self.trans_block4(self.dense_block4(x3)) ## 1024, 16, 16

        xx = self.res(x4) ## 1024, 16, 16

        x4_1 = self.atten4(self.trans4(self.dense4(torch.cat((xx, x4), dim=1)))) ## 512, 32, 32
        x3_1 = self.atten3(self.trans3(self.dense3(torch.cat((x4_1, x3), dim=1)))) ## 256, 64, 64
        x2_1 = self.atten2(self.trans2(self.dense2(torch.cat((x3_1, x2), dim=1))))  ## 128, 128, 128
        x1_1 = self.atten1(self.trans1(self.dense1(torch.cat((x2_1, x1), dim=1))))  ## 64, 256, 256

        out = self.post(x1_1)

        return out

class NetD(nn.Module):
    def __init__(self, nc=3, nf=36):
        super(NetD, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nc, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 36 x 512 x 512
            nn.LeakyReLU(0.2, inplace=True),
            DBlock(nf, nf * 2),  # 72 x 256 x 256
            DBlock(nf * 2, nf * 4),  # 144 x 128 x 128
            DBlock(nf * 4, nf * 8),  # 288 x 64 x 64
            DBlock(nf * 8, nf * 8),  # 288 x 32 x 632
            nn.Conv2d(nf * 8, nf * 8, 4, 1, 1, bias=False),  # 288 x 31 x 31
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, 1, 4, 1, 1, bias=False),  # 288 x 30 x 30
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output

    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req


class DBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        output = self.main(x)
        return output




