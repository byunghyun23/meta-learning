import torch.nn as nn
from torchmeta.modules import MetaBatchNorm2d, MetaConv2d, MetaLinear, MetaModule, MetaSequential


class MetaModel(MetaModule):
    def __init__(self, in_channels: int, out_features: int) -> None:
        super(MetaModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = 64

        self.conv = MetaSequential(
            self.conv_block(self.in_channels, self.hidden_size, 3),
            self.conv_block(self.hidden_size, self.hidden_size, 3),
            self.conv_block(self.hidden_size, self.hidden_size, 3),
            self.conv_block(self.hidden_size, self.hidden_size, 3),
        )

        self.linear = MetaLinear(self.hidden_size, self.out_features)

    @classmethod
    def conv_block(cls, in_channels, out_channels, kernel_size):
        return MetaSequential(
            MetaConv2d(in_channels, out_channels, kernel_size, padding=1, stride=3),
            MetaBatchNorm2d(out_channels, momentum=1.0, track_running_stats=False),
            nn.ReLU(),
        )

    def forward(self, x, params=None):
        convs = self.conv(x, params=self.get_subdict(params, 'conv'))
        prob = self.linear(convs.flatten(start_dim=1), params=self.get_subdict(params, 'linear'))

        return prob