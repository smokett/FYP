import torch
import torch.nn as nn
import torchvision
from torch.nn.init import normal, constant
from basic_ops import SegmentConsensus

class OpticalFlowNet(nn.Module):
    def __init__(self, dropout=0.5, sample_len=2*5):
        super(OpticalFlowNet, self).__init__()
        self.new_length = 5
        self.dropout = dropout
        self.sample_len = sample_len
        self.alexnet = getattr(torchvision.models, "alexnet")(True)
        self.alexnet.last_layer_name = 'classifier'
        self.alexnet = self._construct_flow_model(self.alexnet)
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
    
    def _construct_flow_model(self, base_model):
        modules = list(self.alexnet.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                                conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                                bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        setattr(container, layer_name, new_conv)
        return base_model


    def forward(self, input):
        base_out = self.alexnet(input.view((-1, self.sample_len) + input.size()[-2:]))
        base_out = self.new_fc(base_out)
        base_out = base_out.view((-1, 3) + base_out.size()[1:])
        f1 = self.consensus.apply(base_out)
        f1 = torch.squeeze(f1)
        return f1