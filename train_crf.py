"""
Training loop for conditional random field.

Credit to PyTorch's advanced NLP tutorial for crucial insight into good ways of structuring
a BiLSTM-based CRF code in PyTorch:

http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
"""
import torch
from torch.autograd import Variable
import torch.optim as optim
from crf import ConvCRF, BiLSTMCRF
from data.loader import NPZLoader

##################################################
# Specify all hyperparameters and construct
# model and SGD optimizer.
##################################################
input_dim = 6
embed_dim = 400
num_labels = 7
start_ix = 5
stop_ix = 6
pad_ix = 4
batch_size = 32

# ----- uncomment to use convolutional CRF:
"""
conv_layers = [(1, 400, 512), (3, 512, 512), (3, 512, 512), (3, 512, 512), (3, 512, 512)]
crf = ConvCRF(input_dim, embed_dim, conv_layers, num_labels, batch_size,
              start_token=start_ix, stop_token=stop_ix, gap_token=pad_ix,
              use_cuda=False)
"""
# ----- uncomment to use bidir. lstm CRF:
hidden_dim = 100
assert (hidden_dim % 2 == 0)
crf = BiLSTMCRF(input_dim, embed_dim, hidden_dim, num_labels, batch_size,
                start_token=start_ix, stop_token=stop_ix, gap_token=pad_ix, use_cuda=False)

learning_rate = 0.01
wd = 1e-4
opt = optim.RMSprop(crf.parameters())

##################################################
# Train for 40 epochs; if Ctrl-C interrupts,
# print message and deallocate.
##################################################
num_epochs = 40
print_every = 100
step = 0
try:
    for _ in range(num_epochs):
        dataset = NPZLoader("./data/train_60k.npz",
                            "./data/test_40k.npz",
                            batch_size=batch_size)
        for events, bases in dataset.training_batches:
            # gradients are accumulated by PyTorch; clear them out before each backprop step:
            crf.zero_grad()
    
            # preprocess events and bases batches:
            _events = Variable(events.transpose(0,1).contiguous())
            _bases = Variable(bases.transpose(0,1).contiguous().long())

            # forward pass and compute -CLL(bases|events):
            neg_log_likelihood = crf.neg_log_likelihood(_events, _bases)
            
            # print to stdout occasionally:
            if step % print_every == 0:
                val_events, val_bases = dataset.fetch_test_batch()
                val_events = Variable(val_events.transpose(0,1).contiguous())
                val_bases = Variable(val_bases.transpose(0,1).contiguous().long())
                validation_nll = crf.neg_log_likelihood(val_events, val_bases)
                print("Step: {0} | Training NCLL: {1} | Validation NCLL: {2}".format(
                      step, neg_log_likelihood.data[0], validation_nll.data[0]))

            # compute loss, grads, updates:
            neg_log_likelihood.backward()
            opt.step()

            step += 1

        del dataset
except KeyboardInterrupt:
    del dataset
    print("-" * 80)
    print("Halted training.")

