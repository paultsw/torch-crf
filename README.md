README --- Linear Chain Conditional Random Field
================================================

An implementation of a linear chain CRF in PyTorch.

Usage
-----
To train a model, run `python train_crf.py`.

To set custom training parameters, create a `config.json` file (we recommend that you do this in the `configs/`
folder for record-keeping). Then, run: `python train_crf.py --config config/path/to/config.json`.

Features
--------
There are two types of features: the transition features between different hidden states given by a transitions table,
and features from observations to each hidden timestep given by either a stack of convolutional layers or by a stack
of bidirectional LSTMs.

Dynamic Algorithms
------------------
* Viterbi algorithm
* Forward Algorithm

References
----------
* Sutton, McCallum, _An Introduction to Conditional Random Fields_, `https://arxiv.org/abs/1011.4088`


Credits
-------
Credit to Robert Guthrie's PyTorch's NLP tutorial for insight into good ways of structuring
a BiLSTM-based CRF code in PyTorch:
`http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html`

The functions in `dynamic.py` are also variations of code contained in the above tutorial.
