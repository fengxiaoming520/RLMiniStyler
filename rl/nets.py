import torch
import torch.nn as nn
from function import adaptive_instance_normalization as adain


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groupnum):
        super(ConvLayer, self).__init__()
        # Padding Layer
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groupnum)

    def forward(self, x, weight=None, bias=None, filterMod=False):
        x = self.reflection_pad(x)
        x1 = self.conv_layer(x)
        if filterMod:
            x1 = weight * x1 + bias * x

        return x1


class Upsample(nn.Module):
    def __init__(self, in_planes=64):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = ConvLayer(int(in_planes * slim_factor), int(in_planes//2 * slim_factor), kernel_size=3, stride=1, groupnum=int(in_planes//2 * slim_factor))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(int(in_planes//2 * slim_factor), int(in_planes//2 * slim_factor), kernel_size=1, stride=1, groupnum=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, w, b, filterMod=False):
        x = self.up(x)
        x = self.relu1(self.conv1(x, w, b, filterMod=False))
        x = self.relu2(self.conv2(x, w, b, filterMod))

        return x


class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, groupnum=1):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1, groupnum=groupnum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1, groupnum=groupnum)

    def forward(self, x, weight=None, bias=None, filterMod=False):
        if filterMod:
            x1 = self.conv1(x)
            x2 = weight * x1 + bias * x

            x3 = self.relu(x2)
            x4 = self.conv2(x3)
            x5 = weight * x4 + bias * x3
            return x + x5
        else:
            return x + self.conv2(self.relu(self.conv1(x)))


class ConvDistLayer(nn.Module):
    def __init__(self, in_channels):
        super(ConvDistLayer, self).__init__()
        self.logvar_max = 2
        self.logvar_min = -10
        self.relu = nn.ReLU(inplace=True)
        self.conv_layer = ConvLayer(int(in_channels * slim_factor), int(in_channels * slim_factor), kernel_size=3,
                                    stride=1, groupnum=int(in_channels * slim_factor))
        self.mean = ConvLayer(int(in_channels * slim_factor), int(in_channels * slim_factor), kernel_size=1, stride=1,
                              groupnum=1)
        self.std = ConvLayer(int(in_channels * slim_factor), int(in_channels * slim_factor), kernel_size=1, stride=1,
                             groupnum=1)

    def forward(self, x):
        x = self.conv_layer(x)
        mu = self.mean(x)
        logvar = self.std(x)
        logvar = torch.tanh(logvar)
        logvar = self.logvar_min + 0.5 * (self.logvar_max - self.logvar_min) * (logvar + 1)
        var = logvar.exp()
        dist = torch.distributions.Normal(mu, var)
        return dist


# Control the number of channels
slim_factor = 1


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc1_1 = nn.Sequential(
            ConvLayer(3, int(16 * slim_factor), kernel_size=9, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
        )
        self.enc1_2 = nn.Sequential(
            ConvLayer(int(16 * slim_factor), int(32 * slim_factor), kernel_size=3, stride=2,
                      groupnum=int(16 * slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32 * slim_factor), int(32 * slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
        )
        self.style1_2 = nn.Sequential(
            ConvLayer(int(32 * slim_factor), int(32 * slim_factor), kernel_size=3, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
        )
        self.modulator1_2 = Modulator2(int(32 * slim_factor))
        self.enc1_3 = nn.Sequential(
            ConvLayer(int(32 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=2,
                      groupnum=int(32 * slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
        )
        self.style1_3 = nn.Sequential(
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
        )
        self.modulator1_3 = Modulator2(int(64 * slim_factor))
        self.enc1_4 = nn.Sequential(
            ResidualLayer(int(64 * slim_factor), kernel_size=3),
        )
        self.enc2 = nn.Sequential(
            ResidualLayer(int(64 * slim_factor), kernel_size=3)
        )

    def forward(self, x, style_mode=False):
        w1_2, b1_2, w1_3, b1_3 = None, None, None, None
        x1_1 = self.enc1_1(x)
        x1_2 = self.enc1_2(x1_1)
        if style_mode:
            x1_2 = self.style1_2(x1_2)
            w1_2, b1_2 = self.modulator1_2(x1_2)
        x1_3 = self.enc1_3(x1_2)
        if style_mode:
            x1_3 = self.style1_3(x1_3)
            w1_3, b1_3 = self.modulator1_3(x1_3)
        x1 = self.enc1_4(x1_3)
        x2 = self.enc2(x1)

        out = [x1, x2, x1_2, x1_3, w1_2, b1_2, w1_3, b1_3]
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec1 = ResidualLayer(int(64 * slim_factor), kernel_size=3)
        self.dec2 = ResidualLayer(int(64 * slim_factor), kernel_size=3)

        self.dec3 = Upsample(in_planes=64)

        self.dec4 = Upsample(in_planes=32)
        self.dec5 = nn.Sequential(
            ConvLayer(int(16 * slim_factor), 3, kernel_size=9, stride=1, groupnum=1)
        )

    def forward(self, action, encs, alpha=1.0):
        _, s, w, b = encs
        x1 = adain(action, s[1])
        x1 = alpha * x1 + (1 - alpha) * action

        x2 = self.dec1(x1, w[1], b[1], filterMod=True)

        x3 = adain(x2, s[0])
        x3 = alpha * x3 + (1 - alpha) * x2

        x4 = self.dec2(x3, w[0], b[0], filterMod=True)
        x5 = self.dec3(x4, s[6], s[7], filterMod=True)
        x6 = self.dec4(x5, s[4], s[5], filterMod=True)
        out = self.dec5(x6)

        return out


class Modulator(nn.Module):
    def __init__(self):
        super(Modulator, self).__init__()
        self.weight1 = nn.Sequential(
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=1,
                      groupnum=int(64 * slim_factor)),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.bias1 = nn.Sequential(
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=1,
                      groupnum=int(64 * slim_factor)),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.weight2 = nn.Sequential(
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=1,
                      groupnum=int(64 * slim_factor)),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.bias2 = nn.Sequential(
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=1,
                      groupnum=int(64 * slim_factor)),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        w1 = self.weight1(x[0])
        b1 = self.bias1(x[0])

        w2 = self.weight2(x[1])
        b2 = self.bias2(x[1])

        return [w1, w2], [b1, b2]


class Modulator2(nn.Module):
    def __init__(self, in_planes=64):
        super(Modulator2, self).__init__()
        self.weight1 = nn.Sequential(
            ConvLayer(int(in_planes * slim_factor), int(in_planes//2 * slim_factor), kernel_size=3, stride=1,
                      groupnum=int(in_planes//2 * slim_factor)),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.bias1 = nn.Sequential(
            ConvLayer(int(in_planes * slim_factor), int(in_planes//2 * slim_factor), kernel_size=3, stride=1,
                      groupnum=int(in_planes//2 * slim_factor)),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        w1 = self.weight1(x)
        b1 = self.bias1(x)

        return w1, b1


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.encoder = Encoder()
        self.modulator = Modulator()
        self.dist_builder = ConvDistLayer(int(64 * slim_factor))

    def forward(self, moving, style, alpha=1.):
        style_feats = self.encoder(style, style_mode=True)
        filter_weights, filter_biases = self.modulator(style_feats)

        moving_feats = self.encoder(moving)
        dist = self.dist_builder(moving_feats[1])
        encs = [moving_feats, style_feats, filter_weights, filter_biases]
        return dist, encs

    def sample(self, moving, style, alpha=1.0):
        dist, encs = self.forward(moving, style, alpha)
        action = dist.rsample()
        log_pi = dist.log_prob(action)
        mean = dist.mean
        return action, log_pi, mean, encs


class QNetwork(nn.Module):
    def __init__(self, n_channel=3, nf=16):
        super(QNetwork, self).__init__()
        self.state1 = nn.Sequential(
            ConvLayer(n_channel, int(nf * slim_factor), kernel_size=9, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
        )
        self.state2 = nn.Sequential(
            ConvLayer(int(nf * slim_factor), int(2 * nf * slim_factor), kernel_size=3, stride=2,
                      groupnum=int(nf * slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(2 * nf * slim_factor), int(2 * nf * slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
        )
        self.state3 = nn.Sequential(
            ConvLayer(int(2 * nf * slim_factor), int(4 * nf * slim_factor), kernel_size=3, stride=2,
                      groupnum=int(2 * nf * slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(4 * nf * slim_factor), int(4 * nf * slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
        )
        self.mix = nn.Sequential(
            ConvLayer(int(8 * nf * slim_factor), int(8 * nf * slim_factor), kernel_size=3, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(8 * nf * slim_factor), int(4 * nf * slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, moving, action):  
        x = self.state1(moving)  
        x = self.state2(x) 
        x = self.state3(x)
        x = torch.cat([x, action], dim=1)  
        x = self.mix(x)  

        return x



