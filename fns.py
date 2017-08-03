"""
Helper functions for PyTorch.
"""
import torch
import torch.nn.functional as F
import random


def softmax(inp, axis=1):
    """
    SoftMax function with axis argument. Credit to Yuanpu Xie at:
    https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637/2
    """
    # get size of input:
    input_size = inp.size()

    # transpose dimensions:
    trans_input = inp.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    # form 2d (...,axis) tensor:
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    
    # compute softmax on 2d tensor:
    soft_max_2d = F.softmax(input_2d)
    
    # reshape to original size:
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)


def random_pick(batch1, batch2, proba=0.5):
    """
    Flip a weighted coin and return either batch1 or batch2 based on the result.

    Args:
    * batch1: tensor of any type/shape, as long as it is the same as batch2.
    * batch2: tensor of any type/shape, as long as it is the same as batch1.
    * proba: probability of choosing 
    """
    if random.random() < proba:
        return batch1
    else:
        return batch2


def zero_pad_rank3(tsr, padding, axis=1, use_cuda=False):
    """
    Pad a rank-3 tensor with zeros along an axis.

    Args:
    * tsr: of shape [ax1, ax2, ax3].
    * padding: python integer tuple (pad_start, pad_end) that indicates the number of zeros to add to the front
      and back of some axis of `tsr`.
    * axis: axis along which to pad.
    * use_cuda: whether to allocate tensors to CUDA.
    """
    # if no padding, return:
    if padding[0] < 1 and padding[1] < 1:
        return tsr

    # figure out padding shape:
    tsr_shape = tsr.size()
    zero_pad_start_shape = list(tsr_shape)
    zero_pad_stop_shape = list(tsr_shape)
    zero_pad_start_shape[axis] = padding[0]
    zero_pad_stop_shape[axis] = padding[1]
    
    if use_cuda:
        if padding[0] > 0:
            zero_pad_start = torch.autograd.Variable(torch.zeros(zero_pad_start_shape).cuda())
        if padding[1] > 0:
            zero_pad_stop = torch.autograd.Variable(torch.zeros(zero_pad_stop_shape).cuda())
    else:
        if padding[0] > 0:
            zero_pad_start = torch.autograd.Variable(torch.zeros(zero_pad_start_shape))
        if padding[1] > 0:
            zero_pad_stop = torch.autograd.Variable(torch.zeros(zero_pad_stop_shape))

    # concat zeros along axis:
    if padding[0] > 0 and padding[1] > 0:
        _pad_sequence = (zero_pad_start, tsr, zero_pad_stop)
    if padding[0] < 0 and padding[1] > 0:
        _pad_sequence = (tsr, zero_pad_stop)
    if padding[0] > 0 and padding[1] < 0:
        _pad_sequence = (zero_pad_start, tsr)

    return torch.cat(_pad_sequence, dim=axis)


def sequence_norm(tsr, axis=1, eps=1e-8):
    """
    Normalize a 3D tensor across a sequence-dimension.

    (N.B.: event sequences have START/STOP/PAD bits, but these can be averaged without affecting their values
    as they are one-hot indicators.)
    """
    mu = torch.mean(tsr, axis)
    sigma = torch.std(tsr, axis)
    sigma = torch.max(sigma, eps * torch.ones(sigma.size()))

    mu_stack = torch.stack([mu] * tsr.size(axis), axis)
    sigma_stack = torch.stack([sigma] * tsr.size(axis), axis)

    return (tsr - mu_stack) / sigma_stack


def to_scalar(var):
    """
    Converts a variable to a scalar by formatting into a list and
    returning the first element.

    Args:
    * var: instance of autograd.Variable.
    
    Returns:
    * python float of the first entry in `var`.
    """
    return var.view(-1).data.tolist()[0]


def argmax(vec, dim=1):
    """
    Returns the argmax as python list.

    Args:
    * vec: assumed to be a (variable/tensor) of shape (batch, dims). The argmax is taken along the `dim` axis.
    
    Returns:
    * a python list of indices.
    """
    # compute maximum along dimension 1
    _, ix = torch.max(vec, dim)
    return ix.view(-1).data.tolist()


def log_sum_exp(vec):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm.

    The naive way of computing torch.log(torch.sum(torch.exp(vec))) is numerically unstable
    due to the possibility that torch.exp() will overflow on large values (e.g. 1000).
    This computes it in a more stable way by shifting it down by max(vec) to avoid overflows
    and gracefully handle underflows.

    This implementation works for a batch of vectors.

    Args:
    * vec: assumed to be a Variable of a batch of vectors of shape (batch, num_dim).
    
    Return:
    * logsumexp: FloatTensor Variable of size (batch).
    """
    max_ixs = torch.autograd.Variable(torch.LongTensor(argmax(vec, 1)))
    max_scores = torch.index_select(vec, 1, max_ixs).diag()
    max_scores_broadcast = max_scores.unsqueeze(1).expand(vec.size())
    return max_scores + torch.log(torch.sum(torch.exp(vec - max_scores_broadcast),1))
