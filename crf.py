"""
PyTorch model of a conditional random field with convolutional layers providing local features
from the source sequence to the label sequence.

Recall that a conditional random field is a discriminative model that gives the conditional probability as

  Prob(labels|observations) == exp( feat1(labels,obs) + feat2(labels,obs) + ... + featN(labels,obs) ) / Z(obs)

where Z(obs) is the normalizer over all possible labellings. (In this case we assume a sequential modelling task.)

The sum of features within the exponent is also called the "Score", so that:

  Prob(labels|observations) == exp( Score(labels,obs) ) / [exp( Score(labels',obs) ) + ... ]

where the sum in the denominator is over all possible label sequences. This can be thought of as Softmax(Score(-,obs)),
where the softmax ranges over all possible labellings. (A huge and untractable softmax!!)

To make this more tractable, we make the *assumption* that all features are local, and that they can be expressed
by a convolutional neural network, that is:

  Score(labels,obs) == [ exp( ConvNet(labels[t],obs[t]) ) + exp(transitions[label[t+1][t])     \
                           for t in range(sequence_length) ]

Note that the exponents mean we can do everything in log-space, where multiplications (vulnerable to overflows!)
can instead be replaced by addition operations.
"""
import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch.autograd import Variable
from conv_blocks import ResidualConvBlock
from dynamic import forward_alg, score_sequences, viterbi_decode

__CUDA__ = torch.cuda.is_available()

class ConvCRF(nn.Module):
    """
    Definition of Convolutional CRF module. This module is synchronous, i.e. it
    generates a labelling that is equal in length to the input sequence (so you should
    account for not only <START> and <STOP> but also <GAP> tokens in your `num_labels`
    setting).
    """
    def __init__(self, input_dim, embed_dim, conv_layers, num_labels, batch_size,
                 start_token=0, stop_token=1, gap_token=2, use_cuda=__CUDA__):
        """
        Construct modules of a convolutional fully-connected CRF.
        """
        ### call parent initializer:
        super(ConvCRF, self).__init__()

        ### store inputs as attributes:
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.conv_layers = conv_layers
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.start_token = start_token
        self.stop_token = stop_token
        self.gap_token = gap_token
        self.use_cuda = use_cuda

        ### stateful attributes:
        # projection to embedding dimension:
        self.in_proj = nn.Linear(input_dim, embed_dim)
        assert (embed_dim == conv_layers[0][1]) # sanity check
        # convolutional module to generate features:
        _conv_blocks = [ResidualConvBlock(batch_size, bk[1], bk[2], bk[0], use_cuda=use_cuda) for bk in conv_layers]
        self.conv_stack = nn.Sequential(*_conv_blocks)
        # linear projection to dimension over labels:
        self.out_proj = nn.Linear(conv_layers[-1][2], num_labels)
        # initialize target state transition matrix with random values:
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))


    def init_params(self):
        """
        Glorot-normal initialization of all submodules.
        """
        nn_init.xavier_normal(self.in_proj.weight)
        self.in_proj.bias.data.zero_()
        for k in range(len(self.conv_layers)):
            self.conv_stack[k].init_params()
        nn_init.xavier_normal(self.out_proj.weight)
        self.out_proj.bias.data.zero_()


    def forward(self, source_seq):
        """
        Forward-pass for the Module (NOT the same as the forward algorithm _forward_alg(), which computes the normalizing
        constant for a given observation). Takes a sequence, extracts the features, and then runs viterbi decoding.

        Args:
        * source_seq: a raw input sequence; a FloatTensor Variable of size (sequence, batch, in_dim).

        Returns: a tuple (score, labels) where:
        * score: a FloatTensor variable of shape (batch), giving a score for each path.
        * labels: a FloatTensor variable of shape (seq, batch) giving the most likely decoding for each input sequence.
        """
        features = self.get_convolutional_features(source_seq)
        score, labels = viterbi_decode(features, self.transitions, self.batch_size,
                                       self.num_labels, self.start_token, self.stop_token)
        return (score, labels)


    def neg_log_likelihood(self, source_seq, labels):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        Relies on other methods in this class: _get_convolutional_features(), _forward_alg(), and _score_sequences().

        Args:
        * source_seq: a raw input sequence of shape (seq, batch, in_dim).
        * labels: a sequence of labels (seq, batch) where each value is a long in [0,num_labels).
          The ground truth labels for each respective sequence.

        Returns:
        * a FloatTensor variable of shape (batch) containing scores of each labelling.
        """
        features = self.get_convolutional_features(source_seq)
        # normalizing constant:
        normalizer = forward_alg(features, self.transitions, self.batch_size,
                                 self.num_labels, self.start_token, self.stop_token)
        # best possible score, from ground-truth data:
        best_score = score_sequences(features, labels, self.batch_size, self.transitions,
                                     self.start_token, self.stop_token)
        return torch.sum(normalizer - best_score)


    def viterbi_score(self, source_seq, labels):
        """
        Similar to the negative log likelihood, but instead we compute the difference between the viterbi score
        and the label score.
        """
        features = self.get_convolutional_features(source_seq)
        # score from the viterbi algorithm:
        viterbi_score, _ = viterbi_decode(features, self.transitions, self.batch_size,
                                       self.num_labels, self.start_token, self.stop_token)
        # best possible score, from a ground-truth labelling:
        best_score = score_sequences(features, labels, self.batch_size, self.transitions,
                                     self.start_token, self.stop_token)
        return torch.sum(viterbi_score - best_score)


    def get_convolutional_features(self, source_seq):
        """
        Generate features by passing an input sequence through the internal convolutional
        module.

        Args:
        * source_seq: FloatTensor variable of shape (sequence, batch, in_dim)

        Returns:
        * features_seq: FloatTensor variable of shape (sequence, batch, num_labels).
        """
        conv_in_dim = self.conv_layers[0][1]
        conv_out_dim = self.conv_layers[-1][2]
        embedded_seq = self.in_proj(source_seq.view(-1, self.input_dim)).view(-1, self.batch_size, conv_in_dim)
        conv_out = self.conv_stack(embedded_seq)
        features_seq = self.out_proj(conv_out.view(-1, conv_out_dim)).view(-1, self.batch_size, self.num_labels)
        return features_seq



# ===== ===== ===== ===== BiLSTM CRF Module Implementation:
class BiLSTMCRF(nn.Module):
    """
    Definition of a CRF with features coming from a bidirectional LSTM.
    """
    def __init__(self, input_dim, embed_dim, hidden_dim, num_labels, batch_size,
                 start_token=0, stop_token=1, gap_token=2, use_cuda=__CUDA__):
        """
        Construct modules of a convolutional fully-connected CRF.
        """
        ### call parent initializer:
        super(BiLSTMCRF, self).__init__()

        ### store inputs as attributes:
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.start_token = start_token
        self.stop_token = stop_token
        self.gap_token = gap_token
        self.use_cuda = use_cuda

        ### stateful attributes:
        # projection to embedding dimension:
        self.in_proj = nn.Linear(input_dim, embed_dim)
        # create BiLSTM:
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # linear projection to dimension over labels:
        self.out_proj = nn.Linear(hidden_dim, num_labels)
        # initialize target state transition matrix with random values:
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))


    def init_hidden(self):
        """
        Return initial states for BiLSTM hidden units.
        """
        return (Variable(torch.randn(2, self.batch_size, self.hidden_dim // 2)),
                Variable(torch.randn(2, self.batch_size, self.hidden_dim // 2)))


    def get_bilstm_features(self, source_seq):
        """
        Generate features by passing an input sequence through the internal bidir. LSTM
        module.

        Args:
        * source_seq: FloatTensor variable of shape (sequence, batch, in_dim)

        Returns:
        * lstm_feats: FloatTensor variable of shape (sequence, batch, num_labels).
        """
        source_proj = self.in_proj(source_seq.view(-1, self.input_dim)).view(len(source_seq), self.batch_size, self.embed_dim)
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(source_proj, self.hidden)
        lstm_feats = self.out_proj(lstm_out.view(-1, self.hidden_dim)).view(len(lstm_out), self.batch_size, self.num_labels)
        
        return lstm_feats


    def init_params(self):
        """
        Glorot-normal initialization of all submodules.
        """
        nn_init.xavier_normal(self.in_proj.weight)
        self.in_proj.bias.data.zero_()
        nn_init.xavier_normal(self.out_proj.weight)
        self.out_proj.bias.data.zero_()


    def forward(self, source_seq):
        """
        Forward-pass for the Module (NOT the same as the forward algorithm _forward_alg(), which computes the normalizing
        constant for a given observation). Takes a sequence, extracts the features, and then runs viterbi decoding.

        Args:
        * source_seq: a raw input sequence; a FloatTensor Variable of size (sequence, batch, in_dim).

        Returns: a tuple (score, labels) where:
        * score: a FloatTensor variable of shape (batch), giving a score for each path.
        * labels: a FloatTensor variable of shape (seq, batch) giving the most likely decoding for each input sequence.
        """
        features = self.get_bilstm_features(source_seq)
        score, labels = viterbi_decode(features, self.transitions, self.batch_size,
                                       self.num_labels, self.start_token, self.stop_token)
        return (score, labels)


    def neg_log_likelihood(self, source_seq, labels):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.

        Args:
        * source_seq: a raw input sequence of shape (seq, batch, in_dim).
        * labels: a sequence of labels (seq, batch) where each value is a long in [0,num_labels).
          The ground truth labels for each respective sequence.

        Returns:
        * a FloatTensor variable of shape (batch) containing scores of each labelling.
        """
        features = self.get_bilstm_features(source_seq)
        # normalizing constant:
        normalizer = forward_alg(features, self.transitions, self.batch_size,
                                 self.num_labels, self.start_token, self.stop_token)
        # best possible score, from ground-truth data:
        best_score = score_sequences(features, labels, self.batch_size, self.transitions,
                                     self.start_token, self.stop_token)
        return torch.sum(normalizer - best_score)


    def viterbi_score(self, source_seq, labels):
        """
        Similar to the negative log likelihood, but instead we compute the difference between the viterbi score
        and the label score.
        """
        features = self.get_bilstm_features(source_seq)
        # score from the viterbi algorithm:
        viterbi_score, _ = viterbi_decode(features, self.transitions, self.batch_size,
                                          self.num_labels, self.start_token, self.stop_token)
        # best possible score, from a ground-truth labelling:
        best_score = score_sequences(features, labels, self.batch_size, self.transitions,
                                     self.start_token, self.stop_token)
        return torch.sum(viterbi_score - best_score)
