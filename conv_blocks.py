"""
Convolutional block modules.
"""
import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch.autograd import Variable
import torch.nn.functional as F
from math import floor, ceil
from fns import zero_pad_rank3

_HAS_GPU_ = torch.cuda.is_available()

# ===== ===== ===== ===== Residual Convolutional Block with GLU Activations
class ResidualConvBlock(nn.Module):
    """
    A convolutional block with residual connections.

    (At this current moment, we only support stride == 1 due to the complications of padding calculations.)
    """
    def __init__(self, batch_size, in_channels, out_channels, kwidth, stride=1, use_cuda=_HAS_GPU_):
        """
        Construct ResConvBlock.
        """
        # run parent constructor:
        super(ResidualConvBlock, self).__init__()

        # raise error if stride != 1:
        if stride != 1:
            raise Exception("[NANOSEQ] ERR: strides =/= 1 are not supported.")

        # calculate padding: padding is specified as (pleft, pright, ptop, pbottom, pfront, pback)
        if kwidth % 2 == 0:
            #_pad = (int(floor((kwidth-1) / 2)), int(ceil((kwidth-1) / 2)))
            raise Exception("[NANOSEQ] ERR: even-sized kernel widths not supported.")
        else:
            _pad = (int(ceil((kwidth-1) / 2)), int(ceil((kwidth-1) / 2)))

        # store attributes:
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwidth = kwidth
        self.stride = stride
        self.padding = _pad
        self.use_cuda = use_cuda

        # construct submodules:
        self.conv1d_glu = nn.Conv1d(in_channels, out_channels, kwidth, stride=stride)
        self.conv1d_id = nn.Conv1d(in_channels, out_channels, kwidth, stride=stride)
        self.residual_proj = nn.Linear(in_channels, out_channels)
        self.glu = GLU()

        # move to CUDA if necessary:
        if use_cuda:
            self.conv1d_glu.cuda()
            self.conv1d_id.cuda()
            self.residual_proj.cuda()


    def forward(self, in_seq):
        """
        Forward pass through conv block. Reshape first to (batch, in_depth, seq).

        Args:
        * in_seq: input sequence of shape (seq, batch, in_depth). Pass the sequence through
          convolution blocks and end with a GLU.

        Returns:
        * out_seq: sequence of shape (seq, batch, out_depth).
        """
        # reshape to expected dimensions and apply convolutions:
        in_seq_padded = zero_pad_rank3(in_seq, self.padding, axis=0, use_cuda=self.use_cuda)
        in_seq_reshaped = torch.transpose(torch.transpose(in_seq_padded, 0, 1), 1, 2).contiguous()

        conv_glu = self.conv1d_glu(in_seq_reshaped)
        conv_id = self.conv1d_id(in_seq_reshaped)

        # reshape convolutional outputs:
        conv_glu_reshaped = torch.transpose(torch.transpose(conv_glu, 1, 2), 0, 1).contiguous()
        conv_id_reshaped = torch.transpose(torch.transpose(conv_id, 1, 2), 0, 1).contiguous()

        # apply GLU() and add input sequence:
        glu_out = self.glu(conv_glu_reshaped, conv_id_reshaped)
        in_seq_proj = self.residual_proj(in_seq.view(-1, self.in_channels)).view(
            -1, self.batch_size, self.out_channels)
        return torch.add(glu_out, in_seq_proj)


    def init_params(self):
        """
        Initialize all parameters.
        """
        ### Conv1d-GLU:
        nn_init.xavier_normal(self.conv1d_glu.weight)
        self.conv1d_glu.bias.data.zero_()

        ### Conv1d-Id:
        nn_init.xavier_normal(self.conv1d_id.weight)
        self.conv1d_id.bias.data.zero_()

        ### linear residual projection:
        nn_init.xavier_normal(self.residual_proj.weight)
        self.residual_proj.bias.data.zero_()


    def __repr__(self):
        return self.__class__.__name__ + ' ()'


# ===== ===== ===== ===== GLU Activation:
class GLU(nn.Module):
    """
    A stateless GLU operation.

    Instead of accepting an even-dimensional block, this accepts two blocks of the same shape
    and performs the GLU operation on them.
    """
    def forward(self, x, y):
        return torch.mul(x,F.sigmoid(y))

    def __repr__(self):
        return self.__class__.__name__ + ' ()'
