from torch import nn
import torch

from .network_blocks import BaseConv, Focus, DWConv, BaseConv, ShuffleV2DownSampling, ShuffleV2Basic

class ShuffleNetModified(nn.Module):
    def __init__(
        self,
        channels=[16,32,64],
        out_features=("stage2", "stage3", "stage4"),
        act="silu",
    ):
        super().__init__()
        stage_unit_repeat = [3, 7 ,3]
        self.channels = []
        self.out_features = out_features
        base_channels = channels
        # print(chann)

        self.stem_list = []
        self.stem = nn.Sequential(*self.stem_list)
        
        self.conv1 = DWConv(1, 8, ksize=3,stride=2,act=act)
        self.conv2 = DWConv(8, 8, ksize=3,stride=2,act=act)
        self.stage2_list = [ShuffleV2DownSampling(8, base_channels[0], act=act)]
        for _ in range(stage_unit_repeat[0]):
            self.stage2_list.append(ShuffleV2Basic(base_channels[0], base_channels[0], act=act))
        self.stage2 = nn.Sequential(*self.stage2_list)

        self.stage3_list = [ShuffleV2DownSampling(base_channels[0], base_channels[1],act=act)]
        for _ in range(stage_unit_repeat[1]):
            self.stage3_list.append(ShuffleV2Basic(base_channels[1], base_channels[1], act=act))
        self.stage3 = nn.Sequential(*self.stage3_list)

        self.stage4_list = [ShuffleV2DownSampling(base_channels[1], base_channels[2], act=act)]
        for _ in range(stage_unit_repeat[2]):
            self.stage2_list.append(ShuffleV2Basic(base_channels[2], base_channels[2], act=act))
        self.stage4 = nn.Sequential(*self.stage4_list)
        self.conv5 = BaseConv(base_channels[2], 1024, ksize=4,stride=2,act=act)
        self.conv6 = BaseConv(1024,1,ksize=1,act=act)

    def forward(self, x):
        outputs = {}
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        
        return x
    