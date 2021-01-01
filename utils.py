import os
import random
import multiprocessing

import numpy as np
import torch


def fix_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()

    # https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value


def pad_sequence(sequence_batch, max_seq_length, pad_token_id):
    max_batch_len = max(len(x) for x in sequence_batch)
    max_len = min(max_seq_length, max_batch_len)

    padded_sequences = []
    attention_masks = []

    attend, no_attend = 1, 0

    for sequence in sequence_batch:
        new_sequence = list(sequence[:max_len])
        attention_mask = [attend] * len(new_sequence)

        pad_length = max_len - len(new_sequence)

        new_sequence.extend([pad_token_id] * pad_length)
        attention_mask.extend([no_attend] * pad_length)

        padded_sequences.append(new_sequence)
        attention_masks.append(attention_mask)

    padded_sequences = torch.tensor(padded_sequences)
    attention_masks = torch.tensor(attention_masks)

    return padded_sequences, attention_masks
