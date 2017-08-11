"""
Training loop for conditional random field.
"""
import torch
from torch.autograd import Variable
import torch.optim as optim
from crf import ConvCRF, BiLSTMCRF
from data.loader import NPZLoader
from utils.logging import Logger
import os

##################################################
# Read configuration file from input.
##################################################
from utils.config_tools import json_to_config, config_to_json
import argparse
parser = argparse.ArgumentParser(description="Train a linear-chain CRF model.")
parser.add_argument("--config", dest="config", default="./configs/config.json")
args = parser.parse_args()
config = json_to_config(args.config)
config_to_json(config,os.path.join(config['save_dir'], "config.json"))

##################################################
# Specify all hyperparameters; construct
# model and SGD optimizer.
##################################################
# ----- model settings:
input_dim = config['input_dim']
embed_dim = config['embed_dim']
num_labels = config['num_labels']
start_ix = config['start_ix']
stop_ix = config['stop_ix']
pad_ix = config['pad_ix']
batch_size = config['batch_size']
cuda = torch.cuda.is_available()

# ----- training settings:
num_epochs = config['num_epochs']
max_iters = config['max_iters']
print_every = config['print_every']
save_every = config['save_every']
print_viterbi = config['print_viterbi']
train_set = config['train_set']
test_set = config['test_set']
save_dir = config['save_dir']

# ----- build CRF:
assert (config['crf']['type'] in ['conv', 'lstm'])
# Convolutional CRF:
if (config['crf']['type'] == 'conv'):
    conv_layers = config['crf']['conv_layers']
    crf = ConvCRF(input_dim, embed_dim, conv_layers, num_labels, batch_size,
                  start_token=start_ix, stop_token=stop_ix, gap_token=pad_ix,
                  use_cuda=cuda)
# Bidirectional LSTM CRF:
if (config['crf']['type'] == 'lstm'):
    hidden_dim = config['crf']['hidden_dim']
    layers = config['crf']['lstm_layers']
    assert (hidden_dim % 2 == 0) # sanity check
    crf = BiLSTMCRF(input_dim, embed_dim, hidden_dim, num_labels, batch_size,
                    start_token=start_ix, stop_token=stop_ix, gap_token=pad_ix,
                    num_layers=layers, use_cuda=cuda)

# ----- optimizer settings:
assert (config['optim']['type'] in ['rmsprop', 'lbfgs'])
learning_rate = config['optim']['learning_rate']
wd = config['optim']['wd']
if config['optim']['type'] == 'rmsprop':
    opt = optim.RMSprop(crf.parameters())
if config['optim']['type'] == 'lbfgs':
    opt = optim.LBFGS(crf.parameters())

def rmsprop_step(e,b):
    opt.zero_grad() # (clear grads)
    # forward pass and compute -CLL(bases|events):
    tr_nll = crf.neg_log_likelihood(e, b)
    tr_nll.backward()
    opt.step()
    return tr_nll

def lbfgs_step(e,b):
    def closure():
        opt.zero_grad()
        # forward pass and compute -CLL(bases|events):
        tr_nll = crf.neg_log_likelihood(e, b)
        tr_nll.backward()
        return tr_nll
    opt.step(closure)
    tr_nll = crf.neg_log_likelihood(e, b)
    return tr_nll

if config['optim']['type'] == 'rmsprop':
    optim_step = rmsprop_step
if config['optim']['type'] == 'lbfgs':
    optim_step = lbfgs_step
##################################################
# Train for several epochs; if Ctrl-C interrupts,
# print message and deallocate.
##################################################
logger = Logger(save_dir)
step = 0
try:
    for _ in range(num_epochs):
        dataset = NPZLoader(train_set, test_set, batch_size=batch_size)
        for events, bases in dataset.training_batches:
    
            # preprocess events and bases batches:
            _events = Variable(events.transpose(0,1).contiguous())
            _bases = Variable(bases.transpose(0,1).contiguous().long())
            if cuda:
                _events = _events.cuda()
                _bases = _bases.cuda()

            # compute loss, grads, updates:
            tr_nll = optim_step(_events, _bases)
            
            # print to stdout occasionally:
            if step % print_every == 0:
                val_events, val_bases = dataset.fetch_test_batch()
                val_events = Variable(val_events.transpose(0,1).contiguous())
                val_bases = Variable(val_bases.transpose(0,1).contiguous().long())
                if cuda:
                   val_events = val_events.cuda()
                   val_bases = val_bases.cuda()
                val_nll = crf.neg_log_likelihood(val_events, val_bases)
                if print_viterbi:
                    vscore, vpaths = crf(val_events)
                    print("Viterbi score:")
                    print(vscore)
                    print("Viterbi paths:")
                    print(vpaths)
                logger.log(step, tr_nll.data[0], val_nll.data[0],
                           tr_nll.data[0]/batch_size, val_nll.data[0]/batch_size)

            # serialize model occasionally:
            if step % save_every == 0: logger.save(step, crf)

            step += 1
            if step > max_iters: raise StopIteration

        del dataset
#--- handle keyboard interrupts:
except KeyboardInterrupt:
    del dataset
    logger.close()
    print("-" * 80)
    print("Halted training; reached {} training iterations.".format(step))
except StopIteration:
    del dataset
    logger.close()
    print("-" * 80)
    print("Finished training; reached max iterations of {}.".format(max_iters))
except Exception as e:
    print("Something went wrong:")
    print(e)
    del dataset
    logger.close()
