import torch
import torch.nn as nn
import torchvision
from torch.nn.init import normal, constant
from basic_ops import SegmentConsensus

class RGBNet(nn.Module):
    def __init__(self, dropout=0.5, sample_len=3*1):
        super(RGBNet, self).__init__()
        self.dropout = dropout
        self.sample_len = sample_len
        self.alexnet = getattr(torchvision.models, "alexnet")(True)
        self.alexnet.last_layer_name = 'classifier'
        self.new_fc = self._prepare_tsn(1)
        self.consensus = SegmentConsensus("avg")

    def _prepare_tsn(self, num_class):
        feature_dim = 256 * 6 * 6
        setattr(self.alexnet, self.alexnet.last_layer_name, nn.Dropout(p=self.dropout))
        self.new_fc = nn.Linear(feature_dim, num_class)
        std = 0.001
        normal(self.new_fc.weight, 0, std)
        constant(self.new_fc.bias, 0)
        return self.new_fc


    def forward(self, input):
        base_out = self.alexnet(input.view((-1, self.sample_len) + input.size()[-2:]))
        base_out = self.new_fc(base_out)
        base_out = base_out.view((-1, 3) + base_out.size()[1:])
        f1 = self.consensus.apply(base_out)
        f1 = torch.squeeze(f1)
        return f1
