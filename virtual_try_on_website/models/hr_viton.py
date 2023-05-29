import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm

from virtual_try_on_website.models.viton_hd import BaseNetwork, MaskNorm, ALIASNorm, ALIASResBlock


class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 'up' 'down' 'same'"

        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
            )
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)

        self.block = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.block(residual))


class ConditionGenerator(nn.Module):
    def __init__(self, input1_nc, input2_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(ConditionGenerator, self).__init__()

        self.ClothEncoder = nn.Sequential(
            ResBlock(input1_nc, ngf, norm_layer=norm_layer, scale='down'),  # 128
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),  # 64
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),  # 32
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 16
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')  # 8
        )

        self.PoseEncoder = nn.Sequential(
            ResBlock(input2_nc, ngf, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')
        )

        self.conv = ResBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, scale='same')

        # in_nc -> skip connection + T1, T2 channel
        self.SegDecoder = nn.Sequential(
            ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
            ResBlock(ngf * 4 * 2 + ngf * 4, ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
            ResBlock(ngf * 4 * 2 + ngf * 4, ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
            ResBlock(ngf * 2 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up'),  # 128
            ResBlock(ngf * 1 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up')  # 256
        )

        self.out_layer = ResBlock(ngf + input1_nc + input2_nc, output_nc, norm_layer=norm_layer, scale='same')

        # Cloth Conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        # Person Conv 1x1
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        self.flow_conv = nn.ModuleList([
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
        ])

        self.bottleneck = nn.Sequential(
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
        )

    def normalize(self, x):
        return x

    def forward(self, input1, input2, upsample='bilinear'):
        E1_list = []
        E2_list = []
        flow_list = []
        # warped_grid_list = []

        # Feature Pyramid Network
        for i in range(5):
            if i == 0:
                E1_list.append(self.ClothEncoder[i](input1))
                E2_list.append(self.PoseEncoder[i](input2))
            else:
                E1_list.append(self.ClothEncoder[i](E1_list[i - 1]))
                E2_list.append(self.PoseEncoder[i](E2_list[i - 1]))

        # Compute Clothflow
        for i in range(5):
            N, _, iH, iW = E1_list[4 - i].size()
            grid = make_grid(N, iH, iW)

            if i == 0:
                T1 = E1_list[4 - i]  # (ngf * 4) x 8 x 6
                T2 = E2_list[4 - i]
                E4 = torch.cat([T1, T2], 1)

                flow = self.flow_conv[i](self.normalize(E4)).permute(0, 2, 3, 1)
                flow_list.append(flow)

                x = self.conv(T2)
                x = self.SegDecoder[i](x)

            else:
                T1 = F.interpolate(T1, scale_factor=2, mode=upsample) + self.conv1[4 - i](E1_list[4 - i])
                T2 = F.interpolate(T2, scale_factor=2, mode=upsample) + self.conv2[4 - i](E2_list[4 - i])

                flow = F.interpolate(flow_list[i - 1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2,
                                                                                                                  3,
                                                                                                                  1)  # upsample n-1 flow
                flow_norm = torch.cat(
                    [flow[:, :, :, 0:1] / ((iW / 2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH / 2 - 1.0) / 2.0)], 3)
                warped_T1 = F.grid_sample(T1, flow_norm + grid, padding_mode='border')

                flow = flow + self.flow_conv[i](
                    self.normalize(torch.cat([warped_T1, self.bottleneck[i - 1](x)], 1))).permute(0, 2, 3, 1)  # F(n)
                flow_list.append(flow)

                x = self.SegDecoder[i](torch.cat([x, E2_list[4 - i], warped_T1], 1))

        N, _, iH, iW = input1.size()
        grid = make_grid(N, iH, iW)

        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
        flow_norm = torch.cat(
            [flow[:, :, :, 0:1] / ((iW / 2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH / 2 - 1.0) / 2.0)], 3)
        warped_input1 = F.grid_sample(input1, flow_norm + grid, padding_mode='border')

        x = self.out_layer(torch.cat([x, input2, warped_input1], 1))

        warped_c = warped_input1[:, :-1, :, :]
        warped_cm = warped_input1[:, -1:, :, :]

        return flow_list, x, warped_c, warped_cm


class SPADEGenerator(BaseNetwork):
    def __init__(self, input_nc):
        super(SPADEGenerator, self).__init__()
        self.fine_width = 768
        self.fine_height = 1024
        self.sh, self.sw = self.compute_latent_vector_size()

        nf = 64
        self.conv_0 = nn.Conv2d(input_nc, nf * 16, kernel_size=3, padding=1)
        for i in range(1, 8):
            self.add_module('conv_{}'.format(i), nn.Conv2d(input_nc, 16, kernel_size=3, padding=1))

        self.head_0 = ALIASResBlock(nf * 16, nf * 16, use_mask_norm=False)

        self.G_middle_0 = ALIASResBlock(nf * 16 + 16, nf * 16, use_mask_norm=False)
        self.G_middle_1 = ALIASResBlock(nf * 16 + 16, nf * 16, use_mask_norm=False)

        self.up_0 = ALIASResBlock(nf * 16 + 16, nf * 8, use_mask_norm=False)
        self.up_1 = ALIASResBlock(nf * 8 + 16, nf * 4, use_mask_norm=False)
        self.up_2 = ALIASResBlock(nf * 4 + 16, nf * 2, use_mask_norm=False)
        self.up_3 = ALIASResBlock(nf * 2 + 16, nf * 1, use_mask_norm=False)
        self.up_4 = ALIASResBlock(nf * 1 + 16, nf // 2, use_mask_norm=False)
        nf = nf // 2

        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def compute_latent_vector_size(self):
        num_up_layers = 7
        sh = self.fine_height // 2**num_up_layers
        sw = self.fine_width // 2**num_up_layers
        return sh, sw

    def forward(self, x, seg):
        samples = [F.interpolate(x, size=(self.sh * 2**i, self.sw * 2**i), mode='nearest') for i in range(8)]
        features = [self._modules['conv_{}'.format(i)](samples[i]) for i in range(8)]

        x = self.head_0(features[0], seg)
        x = self.up(x)
        x = self.G_middle_0(torch.cat((x, features[1]), 1), seg)
        x = self.up(x)
        x = self.G_middle_1(torch.cat((x, features[2]), 1), seg)

        x = self.up(x)
        x = self.up_0(torch.cat((x, features[3]), 1), seg)
        x = self.up(x)
        x = self.up_1(torch.cat((x, features[4]), 1), seg)
        x = self.up(x)
        x = self.up_2(torch.cat((x, features[5]), 1), seg)
        x = self.up(x)
        x = self.up_3(torch.cat((x, features[6]), 1), seg)
        x = self.up(x)
        x = self.up_4(torch.cat((x, features[7]), 1), seg)

        x = self.conv_img(self.relu(x))
        return self.tanh(x)


def make_grid(N, iH, iW):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    grid = torch.cat([grid_x, grid_y], 3)
    return grid