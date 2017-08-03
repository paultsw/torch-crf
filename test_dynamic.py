"""
Unit tests for dynamic programming algorithms.
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from dynamic import forward_alg, score_sequences, viterbi_decode

num_labels = 6
batch_size = 5
gap_token = 0
start_token = 1
stop_token = 2
sequence_length = 17

transitions = nn.Parameter(torch.randn(num_labels, num_labels))
features = Variable(torch.randn(sequence_length, batch_size, num_labels))
labels = Variable(torch.rand(sequence_length, batch_size).mul_(num_labels).clamp(0,num_labels-0.1).long())

print("==================== Forward Algorithm Tests: ====================")
print(forward_alg(features, transitions, batch_size, num_labels, start_token, stop_token))

print("==================== Score Sequences Tests: ====================")
print(score_sequences(features, labels, batch_size, transitions, start_token, stop_token))

print("==================== Viterbi Decoding Tests: ====================")
scores, best_paths = viterbi_decode(features, transitions, batch_size, num_labels, start_token, stop_token)
print("Best path scores:")
print(scores)
print("Best paths:")
print(best_paths)
