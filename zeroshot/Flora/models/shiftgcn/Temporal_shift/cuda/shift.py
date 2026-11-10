from torch.nn import Module, Parameter
from torch.autograd import Function
import sys
import torch
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import shift_cuda
except ImportError:
    # Compiled op not built. Fine when Shift-GCN is never run (use_features=True);
    # calling Shift would fail with a clear error instead of breaking the import.
    class _MissingShiftCuda:
        def __getattr__(self, name):
            raise ImportError("shift_cuda extension is not built (see Flora README to compile it)")
    shift_cuda = _MissingShiftCuda()


import numpy as np 

class ShiftFunction(Function):
    
    @staticmethod
    def forward(ctx, input,xpos,ypos,stride=1):
        if stride==1:
            xpos = xpos
            ypos = ypos
        else:
            ypos = ypos + 0.5
            # ypos = ypos + 0.5
        input = input.contiguous()
        output = shift_cuda.forward(input,xpos,ypos,stride)
        ctx.save_for_backward(input, output, xpos, ypos)
        ctx.stride = stride
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        grad_output = grad_output.contiguous()
        input, output, xpos, ypos = ctx.saved_variables
        grad_input,grad_xpos,grad_ypos = shift_cuda.backward(grad_output, input, output, xpos, ypos, ctx.stride)
        return grad_input, grad_xpos, grad_ypos, None
        
class Shift(Module):

    def __init__(self, channel, stride, init_scale=3):
        super(Shift, self).__init__()

        self.stride = stride

        self.xpos = Parameter(torch.zeros(channel,requires_grad=True,device='cuda')*1.5)
        self.ypos = Parameter(torch.zeros(channel,requires_grad=True,device='cuda')*1.5)

        self.xpos.data.uniform_(-1e-8,1e-8)
        self.ypos.data.uniform_(-init_scale,init_scale)

    def forward(self, input):
        return ShiftFunction.apply(input,self.xpos,self.ypos,self.stride)