import more_itertools

import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

from utils import optimal_num_of_loader_workers, pad_sequence


class DynamicPaddingDataset(Dataset):
    def __init__(self, data, targets=None, workers=None):

        self._data = data
        self._targets = targets
        self._workers = workers if workers else optimal_num_of_loader_workers()

        # to set it while calling get_dataloader
        self.sampler = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if self._targets is not None:
            return self._data[item], self._targets[item]
        else:
            return self._data[item]

    def get_dataloader(self, batch_size, max_length, pad_token_id):

        self.sampler = Sampler(data_source=self._data,
                               batch_size=batch_size)

        collate_fn = Collate(targets=self._targets,
                             max_length=max_length,
                             pad_token_id=pad_token_id)

        dataloader = DataLoader(dataset=self,
                                batch_size=batch_size,
                                sampler=self.sampler,
                                collate_fn=collate_fn,
                                num_workers=self._workers,
                                pin_memory=True)

        return dataloader


class Sampler(Sampler):

    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)

        return self._backsort_inds

    def __init__(self, data_source, batch_size):
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths)

        batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        # in case of empty list
        if batches:
            last_batch = batches.pop(-1)
            np.random.shuffle(batches)
            batches.append(last_batch)

        self._inds = list(more_itertools.flatten(batches))
        self._backsort_inds = None

    def __iter__(self):
        it = iter(self._inds)
        return it

    def __len__(self):
        return len(self._inds)


class Collate:
    def __init__(self, targets, max_length, pad_token_id):
        self._targets = targets
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        if self._targets is not None:
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)

        input_ids, attention_mask = pad_sequence(
            sequences,
            max_seq_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        import pdb
        pdb.set_trace()

        if self._targets is not None:
            output = input_ids, attention_mask, targets
        else:
            output = input_ids, attention_mask

        return output
