"""
Data-loading abstraction class.
"""
import numpy as np
import torch
import torch.utils.data as data_utils

pin_memory = True if torch.cuda.is_available() else False

# ===== ===== NPZLoader: convert NPZ to torch tensor and expose iterator
class NPZLoader(object):
    """
    Load a single (train,test) pair of NPZ files and expose their iterators.
    """
    def __init__(self, train_npz, test_npz,
                 bases_name='references', blengths_name='references_lengths',
                 events_name='events', elengths_name='events_lengths',
                 batch_size=32, drop_last=True):
        """Initialize NPZLoader."""

        ### store arguments:
        self.train_npz = train_npz
        self.test_npz = test_npz
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        ### load datasets:
        _train_set = np.load(train_npz)
        _train_events = torch.from_numpy(_train_set[events_name])
        _train_bases = torch.from_numpy(_train_set[bases_name])
        self.train_set = data_utils.TensorDataset(_train_events, _train_bases)

        _test_set = np.load(test_npz)
        self._test_events = _test_set[events_name]
        self._test_bases = _test_set[bases_name]

        ### create loader/iterator for training set:
        self.training_batches = data_utils.DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, drop_last=drop_last)

    def fetch_test_batch(self):
        """
        Fetch a random test batch; used sparingly for validation purposes, so we don't need a DataLoader
        for this.
        """
        # create selection of random indices without replacement:
        random_ixs = np.random.choice(self._test_events.shape[0], self.batch_size, replace=False)

        # get a random set of events:
        _events = self._test_events[random_ixs, :]
        
        # get a random set of bases (using same random indices):
        _bases = self._test_bases[random_ixs, :]

        # convert to torch:
        events = torch.from_numpy(_events)
        bases = torch.from_numpy(_bases)

        # return pair:
        return (events, bases)


# ===== ===== BasecallDataset: construct a TensorDataset from a single NPZ file and expose fetch()
class BasecallDataset(object):
    def __init__(self, npz_path, bases_name='references', blengths_name='references_lengths',
                 events_name='events', elengths_name='events_lengths',
                 batch_size=32, drop_last=True):
        """
        Construct a new TensorDataset from an NPZ file.
        """
        ### store arguments:
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.drop_last = drop_last

        ### load dataset:
        _dataset = np.load(npz_path)
        self.events = _dataset[events_name]
        self.bases = _dataset[bases_name]


    def fetch(self):
        """
        Fetch a random test batch from the dataset.

        * TODO: make this deterministic.
        """
        # create selection of random indices without replacement:
        random_ixs = np.random.choice(self.events.shape[0], self.batch_size, replace=False)

        # get a random set of events:
        _events = self.events[random_ixs, :]
        
        # get a random set of bases (using same random indices):
        _bases = self.bases[random_ixs, :]

        # convert to torch:
        events = torch.from_numpy(_events)
        bases = torch.from_numpy(_bases)

        # return pair:
        return (events, bases)

# ===== ===== Folder Loader: abstraction to seamlessly iterate over a whole folder's datasets
class FolderLoader(object):
    """
    Load data from a list of NumPy files; has support for multi-process workers and multiple epochs.
    """
    def __init__(self, npz_files_list, batch_size=32, num_epochs=1, shuffle=True, num_workers=1):
        """
        Construct the queue with a number of workers from a list of NumPy files.

        WORK IN PROGRESS. Please use the single-file NPZ loader for now and iterate over
        epochs with a for-loop.

        Implementation references:
        https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py
        https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
        """
        pass


# ===== ===== ===== ===== BatchSampler, imported from PyTorch repository:
class BatchSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

# ===== ===== ===== ===== Example usage of torch.utils.data:
"""
# 1) load from NumPy to torch.Tensor:
_npz = np.load("./datasets/ecoli/train_80pct.npz")
events_tensor = torch.from_numpy(_npz['events'])
bases_tensor = torch.from_numpy(_npz['references'])

# 2) construct training set and dataloader objects:
train_set = data_utils.TensorDataset(events_tensor, bases_tensor)
train_loader = data_utils.DataLoader(train_set, # dataset to load from
                                     batch_size=32, # examples per batch (default: 1)
                                     shuffle=True,
                                     sampler=None, # if a sampling method is specified, `shuffle` must be False
                                     batch_sampler=False, # mutually exclusive with many of the other flags...
                                     num_workers=5, # subprocesses to use for sampling
                                     pin_memory=False, # whether to return an item pinned to GPU
                                     drop_last=True # drop the last batch if not big enough to fit a full batch
)

# 3) can get a single (features,target) pair from dataset via:
_events, _bases = train_set[123] # gets events,bases pair number 123 in the dataset

# 4) can fetch a batch from train_loader via a standard python for-loop:
for event_sequence, base_sequence in train_loader:
    # ... do something with event/base sequence here ...
    pass
"""
