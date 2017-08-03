"""
Generate simulated kmer model based on (mixture of) gaussian distributions.
"""
import numpy as np

# default conversion table for indices
_BASE_TO_INDEX = { 'A': 0, 'G': 1, 'C': 2, 'T': 3, 'PAD': 4, 'START': 5, 'STOP': 6 }

# default conversion table for events: mapping from base indices to a 3-dimensional gaussian RV
# each value is of form (mu_x, mu_y, mu_z, scale)
_INDEX_TO_DIST = {
    0: (0.1, 0.2, 0.3, 1),
    1: (1.0, 0.1, 0.9, 1),
    2: (2.0, 0.1, 3.2, 1),
    3: (3.0, 1.1, 0.3, 1)
}
# A well-separated Gaussian model:
#_INDEX_TO_DIST = {
#    0: (0.1, 4.2, 5.3, 1),
#    1: (1.0, 5.1, 6.9, 1),
#    2: (2.0, 6.1, 7.2, 1),
#    3: (3.0, 7.1, 8.3, 1)
#}

# ===== ===== ===== ===== =====
def generate_dataset(num_examples, base2ix=_BASE_TO_INDEX, event_model=_INDEX_TO_DIST, sequence_range=(20,30)):
    """
    Generate random sequences.

    Args:
    *
    *
    *
    *

    Returns:
    (references, events, references_lengths, events_lengths)
    """
    ### throw error if sequence range is too narrow:
    if (sequence_range[1] - sequence_range[0]) < 4:
        raise Exception("ERROR: sequence range is too narrow. Please use gap of at least 4.")

    ### look up indices:
    _nucleo_ixs = [ int(base2ix['A']), int(base2ix['G']), int(base2ix['C']), int(base2ix['T']) ]
    _start_ix = int(base2ix['START'])
    _stop_ix = int(base2ix['STOP'])
    pad_value = int(base2ix['PAD'])

    ### generate `num_examples` base sequences with random lengths:
    seq_lengths = np.random.randint(low=sequence_range[0], high=(sequence_range[1]-2), size=num_examples, dtype=int)
    base_sequences = []
    for seqlen in seq_lengths:
        base_sequences.append(
            np.array([_start_ix] + np.random.choice(_nucleo_ixs, size=seqlen).tolist() + [_stop_ix]))

    ### sample gaussian events for each base (maybe):
    event_sequences = []
    for bseq in base_sequences:
        event_sequences.append(
            _sample_events_from_base(bseq, _INDEX_TO_DIST, 6, _start_ix, _stop_ix)
        )

    ### pad+concatenate base sequences together:
    padded_base_seqs = []
    for bseq in base_sequences:
        if sequence_range[1] - bseq.shape[0] > 0:
            padded_base_seqs.append(
                np.pad(bseq, (0,(sequence_range[1]-bseq.shape[0])), mode='constant', constant_values=pad_value))
        else:
            padded_base_seqs.append(bseq)
    base_sequences_ = np.stack(padded_base_seqs)

    ### pad+concatenate event sequences together:
    # [TODO: FIX THIS]
    padded_event_seqs = []
    for eseq in event_sequences:
        if sequence_range[1] - eseq.shape[0] > 0:
            _PADVEC = np.array([0,0,0,0,0,1], dtype=np.float32) # event pad vec
            padlength = sequence_range[1] - eseq.shape[0]
            padded_eseq = np.concatenate((eseq, (_PADVEC,) * padlength), axis=0)
            padded_event_seqs.append(padded_eseq)
        else:
            padded_event_seqs.append(eseq)
    event_sequences_ = np.stack(padded_event_seqs)

    ### return bundled tuple of datasets:
    return (base_sequences_, event_sequences_, seq_lengths, seq_lengths)


# ===== ===== ===== ===== ===== base seq => event seq sampler
def _sample_events_from_base(seq, dist_lookup, event_dim, start_ix, stop_ix):
    """
    Take a sequence of nucleotides and sample a 3d event from each one (synchronously).

    (WIP: need to allow variable dimension, random skips/stays, etc.)

    Args:
    * seq:
    * dist_lookup:
    * event_dim:
    * start_ix:
    * stop_ix:
    
    Returns:
    event_seq: shape ( len(seq) , event_dim ), numpy ndarray.
    """
    # perform sampling loop:
    event_samples = []
    for k in range(seq.shape[0]):
        if seq[k] == start_ix:
            event = (0,0,0,1,0,0) # event_start
        elif seq[k] == stop_ix:
            event = (0,0,0,0,1,0) # event_stop
        else:
            # extract one three-dim sample:
            mu_x, mu_y, mu_z, sigma = dist_lookup[seq[k]]
            x = float(np.random.normal(mu_x, sigma))
            y = float(np.random.normal(mu_y, sigma))
            z = float(np.random.normal(mu_z, sigma))
            event = (x,y,z,0,0,0)
        event_samples.append(event)

    return np.array(event_samples, dtype=np.float32)


# ===== ===== ===== ===== ===== generate data when called from cmd line:
if __name__ == "__main__":
    # generate and save training set:
    train_set = generate_dataset(60000)
    np.savez("train_60k.npz", references=train_set[0], events=train_set[1],
             references_lengths=train_set[2], events_lengths=train_set[3]) 

    # generate and save test set:
    test_set = generate_dataset(40000)
    np.savez("test_40k.npz", references=test_set[0], events=test_set[1],
             references_lengths=test_set[2], events_lengths=test_set[3])
