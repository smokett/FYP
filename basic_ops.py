import torch
import math


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    @staticmethod
    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        output = input_tensor.mean(dim=1, keepdim=True)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_in = grad_output.expand(self.shape) / float(self.shape[1])
        return grad_in


