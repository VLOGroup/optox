import numpy as np
import torch
import torch.nn as nn

import unittest

import _ext.th_demosaicing_operator


class DemosaicingFunction(torch.autograd.Function):
    @staticmethod
    def _get_operator(dtype, bayer_pattern):
        if dtype == torch.float32:
            return _ext.th_demosaicing_operator.Demosaicing_float(bayer_pattern)
        elif dtype == torch.float64:
            return _ext.th_demosaicing_operator.Demosaicing_double(bayer_pattern)
        else:
            raise RuntimeError('Unsupported dtype!')

    @staticmethod
    def forward(ctx, x, bayer_pattern):
        ctx.save_for_backward(x)
        ctx.op = DemosaicingFunction._get_operator(x.dtype, bayer_pattern)
        return ctx.op.forward(x)

    @staticmethod
    def backward(ctx, grad_in):
        x = ctx.saved_tensors
        grad_x = ctx.op.adjoint(grad_in)
        return grad_x, None


class Demosaicing(nn.Module):
    def __init__(self, bayer_pattern):
        super(Demosaicing, self).__init__()

        self.bayer_pattern = bayer_pattern

        self.op = DemosaicingFunction

    def forward(self, x):
        # first reshape the input
        x = x.permute(0, 2, 3, 1).contiguous()
        # compute the output
        x = self.op.apply(x_r, self.bayer_pattern)
        return x.transpose_(0, 3, 1, 2)

    def extra_repr(self):
        s = "bayer_pattern={bayer_pattern}"
        return s.format(**self.__dict__)

# TODO: unittests
