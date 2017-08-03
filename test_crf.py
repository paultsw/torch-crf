"""
Unit testing for encoder modules.
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

from crf import ConvCRF

### constructor parameters:
input_dim = 7
embed_dim = 20
conv_layers = [(1,embed_dim,30), (3,30,40), (5,40,40), (3,40,30), (1,30,25)]
num_labels = 8
batch_size = 5
seq_length = 11

### construct convolutional CRF module:
crf = ConvCRF(input_dim, embed_dim, conv_layers, num_labels, batch_size,
              start_token=0, stop_token=1, gap_token=2, use_cuda=False)
crf.init_params()

### create input sequence:
inseq = Variable(torch.randn(seq_length, batch_size, input_dim))

### run conv-crf and print outputs:
score, pred_labels = crf.forward(inseq)
print("===== ===== CRF Score: ===== =====")
print(score.data)
print("===== ===== Predicted Labels: ===== =====")
print(pred_labels)

### create random labelling:
true_labels = Variable(torch.randn(seq_length, batch_size).mul_(num_labels).clamp_(min=0,max=num_labels-1).long())
print("===== ===== Correct Labels: ===== =====")
print(true_labels)

### compute -CLL loss and backprop:
ncll = crf.neg_log_likelihood(inseq, true_labels)
print("===== ===== Negative Conditional Log Likelihood: ===== =====")
print(ncll)


print("All done!")
