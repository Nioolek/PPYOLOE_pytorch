import torch
import torch.nn as nn
import torch.nn.functional as F

from ppyoloe.models.backbone import RepVggBlock
from .network_blocks import get_activation, ConvBNLayer
from .drop import DropBlock2d


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', shortcut=True):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class SPP(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 act='swish',
                 ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(
                kernel_size=size,
                stride=1,
                padding=size // 2,
                ceil_mode=False)
            self.add_module('pool{}'.format(i),
                            pool
                            )
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y


class CSPStage(nn.Module):
    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', spp=False):
        super(CSPStage, self).__init__()

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock':
                self.convs.add_module(str(i), BasicBlock(next_ch_in, ch_mid, act=act, shortcut=False))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp',
                    SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act)
                )
            next_ch_in = ch_mid
        # self.convs = nn.Sequential(*convs)
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], axis=1)
        y = self.conv3(y)
        return y


class CustomCSPPAN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=[1024, 512, 256],
                 norm_type='bn',
                 act='leaky',
                 stage_fn='CSPStage',
                 block_fn='BasicBlock',
                 stage_num=1,
                 block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False,
                 width_mult=1.0,
                 depth_mult=1.0,
                 ):
        super().__init__()
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act = get_activation(act) if act is None or isinstance(act,
                                                               (str, dict)) else act
        self.num_blocks = len(in_channels)
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        self.fpn_stages = nn.ModuleList()
        self.fpn_routes = nn.ModuleList()


        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = nn.Sequential()
            for j in range(stage_num):
                if stage_fn == 'CSPStage':
                    stage.add_module(
                        str(j),
                        CSPStage(block_fn,
                                 ch_in if j == 0 else ch_out,
                                 ch_out,
                                 block_num,
                                 act=act,
                                 spp=(spp and i == 0))
                    )
                else:
                    raise NotImplementedError

            if drop_block:
                stage.append(DropBlock2d(drop_prob=1 - keep_prob, block_size=block_size))
            self.fpn_stages.append(stage)

            if i < self.num_blocks - 1:
                self.fpn_routes.append(
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out // 2,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act))
            ch_pre = ch_out

        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNLayer(
                    ch_in=out_channels[i + 1],
                    ch_out=out_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act))

            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(
                    str(j),
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   spp=False))
            if drop_block:
                stage.add_module('drop', DropBlock2d(block_size, keep_prob))

            pan_stages.append(stage)

        self.pan_stages = nn.Sequential(*pan_stages[::-1])
        self.pan_routes = nn.Sequential(*pan_routes[::-1])

    def forward(self, blocks):
        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], axis=1)
            # route = block
            # for layer in self.fpn_stages[i]:
            #     route = layer(block)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2.)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.cat([route, block], axis=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]
