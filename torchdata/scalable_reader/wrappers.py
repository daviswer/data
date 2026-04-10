import os
import pickle
from collections import deque
from copy import deepcopy
from typing import Any, Callable, Dict, List

import torch

from .base_loader import _StatefulDataset
from .shard_rescaler import atomic_rescale


"""
Implements additional layers of functionality for data loading pipelines built on ScalableReader.

Additional layers are implemented as wrappers for existing pipelines, adding another stage of
preprocessing for each layer (i.e. shuffling, subdataset sampling, etc). Wrappers extend
_StatefulDataset and are compatible with the tagged-state checkpointing framework.

Example usage is as follows:
  data = ScalableReader(path, rank, worldsize, filehandler, delimiter)
  data = SamplingDataset(path, data, delimiter, datasets, weights)
  data = DocPackingDataset(data, seq_len, n_pads, delimiter, pad)
  data = ShuffleDataset(data, buffer_size)
  data = PreProcessDataset(data, lambda x: torch.tensor(x))
  loader = StatefulDataLoader(data, batch_size=1, num_workers=1)

This pipeline loads documents from the specified path, pulling from individual subdatasets according
to specified token ratios, packs and slices the documents into training sequences of length seq_len,
maintains a buffer of buffer_size sequences to perform local shuffling, and finally converts each
data sequence to a torch tensor. It also supports rescalable checkpoint saving and loading.
"""

def _int_tensor(x):
    return torch.tensor(x, device="cpu", dtype=torch.int32)

class _NestedStatefulDataset(_StatefulDataset):
    """
    Stub for nested wrappers of _StatefulDatasets. Extends state fns with recursion.
    Requires a single instantiated sub-dataset (which may be replicated during setup fn).
    The resulting self.dataset must either be a _StatefulDataset, or iterable of _StatefulDatasets.
    If sub-dataset emits a state dict with tag-subdicts that are flat, tag-subdicts for this layer will
    also be flat (ensuring these can be used with the DCP saving/loading functions in dcp_utils.py).
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
    ):
        self.dataset = dataset
        # Inherit default flags from sub-dataset
        super().__init__(self.dataset.datapath, self.dataset.rank, self.dataset.worldsize)

    def setup(self):
        """
        Datapath/rank/worldsize percolate upwards recursively during initialization, so
        now we project any desired changes downward, also recursively.
        We also project local_worldsize downward to prevent subsequent layers from
        further inflating the rank/worldsize - we only need to account for multiprocessing once!
        Any code overriding this function should still include this functionality.
        """
        if not self.is_setup:
            super().setup()
            self.dataset.datapath = self.datapath
            self.dataset.rank = self.rank
            self.dataset.worldsize = self.worldsize
            self.dataset.local_worldsize = self.local_worldsize
            self.dataset.setup()

    def load_state_dict(self, state_dict):
        """
        Sets all specified flags at the current level, then recurses into wrapped dataset.
        If multiple subdatasets are present, uses corresponding key prefixes to retrieve subdicts.
        """
        self.setup()
        super().load_state_dict(state_dict)
        if isinstance(self.dataset, _StatefulDataset):
            self.dataset.load_state_dict(state_dict)
        else:
            for i,subdata in enumerate(self.dataset):
                prefix = self.statename("", i)
                subdict = {state_type:{k[len(prefix):]:v
                                       for k,v in state_dict[state_type].items()
                                       if prefix in k}
                           for state_type in state_dict}
                subdata.load_state_dict(subdict)

    def state_dict(self):
        """
        Fetches state dict recursively from wrapped layers, then adds specified flags.
        Overlapping flags are overwritten by values from THIS layer.
        If multiple subdatasets are present, prepends class name and index to the key string.
        """
        self.setup()
        state = super().state_dict()
        if isinstance(self.dataset, _StatefulDataset):
            substate = self.dataset.state_dict()
            for state_type in state.keys():
                state[state_type].update(substate[state_type])
        else:
            for i,subdata in enumerate(self.dataset):
                substate = subdata.state_dict()
                for state_type in state.keys():
                    state[state_type].update(
                        {self.statename(k,i):v for k,v in substate[state_type].items()}
                    )
        return state


class PreprocessDataset(_NestedStatefulDataset):
    """
    Wrapper for a _StatefulDataset that applies a specified preprocessing
    or augmentation function to dataset outputs.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    aug_fn : function (any -> any)
        The augmentation function to apply to each dataset item.
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
        aug_fn: Callable,
    ):
        super().__init__(dataset)
        self.aug_fn = aug_fn

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            out = next(dataset)
            yield self.aug_fn(out)


class CollateDataset(_NestedStatefulDataset):
    """
    Wrapper for a _StatefulDataset that applies a specified collation function
    to dataset outputs.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    collate_fn : function (List[any] -> any)
        The collation function to apply to each group of dataset items
    batch_size : int
        The number of items expected by the collator fn
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
        collate_fn: Callable[[List[Any]],Any],
        batch_size: int,
    ):
        super().__init__(dataset)
        self.col_fn = collate_fn
        self.bsize = batch_size

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            out = [next(dataset) for _ in range(self.bsize)]
            yield self.col_fn(out)


class ShuffleDataset(_NestedStatefulDataset):
    """
    Wrapper for a StatefulDataset that implements data shuffling via a single in/out buffer.
    Fills buffer two at a time, up to desired size, then switches to one at a time to maintain size.
    Passes randomly sampled outputs one by one.
    Ensures local mixing of data without relying on sliding windows or shuffling of large buffers.
    Any two consecutive inputs will be separated by window_size steps in expectation.
    Rescaling-enabled: buffers that shrink will re-grow to window_size over time, while buffers that
    expand will shrink back down to window_size over time.
    Sequences pulled from the wrapped StatefulDataset must all be constant length.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    window_size : int
        Target size of input/output buffer
    seed : int
        Random seed to use for shuffling
    """

    def __init__(self, dataset: _StatefulDataset, window_size: int, seed: int=42):
        super().__init__(dataset)
        assert (
            window_size > 1
        ), f"Window size {window_size} must be greater than 1 for shuffling to occur"
        self.window_size = window_size
        self.g_state = None
        self.generator = None
        self.buffer: List[List[Any]] = [] # holds the entire data - 10000 - GB - savings is okay, loading + resharding might get tricky
                    # indices 18 - global mapping - NFS - pull up 18.
                    # actual data / rows
                    # TODO: how DCP load works?
        self.buffer_size = 0
        self.state_vars = ["g_state"]
        self.reshard_vars = ["buffer"] # TODO: costs of storing buffer on disk, and resharding it using DCP (comms cost?)
        self.seed = seed

    def setup(self):
        if not self.is_setup:
            super().setup()
            self.generator = torch.Generator().manual_seed(self.rank + self.seed)

    def __iter__(self):
        self.setup()
        dataset = iter(self.dataset)
        # Pad out buffer if needed
        self._pad_buffer()
        first_draw = next(dataset)
        # If buffer entries have wrong length, reset buffer
        if len(first_draw) != len(self.buffer[0]):
            self.buffer = []
            self.buffer_size = 0
            self._pad_buffer()
        while True:
            # If buffer is undersized, add a datapoint
            if self.buffer_size < self.window_size:
                self.buffer[self.buffer_size] = first_draw if first_draw is not None else next(dataset)
                first_draw = None
                self.buffer_size += 1
            # Swap out randomly sampled value from buffer.
            i = torch.randint(self.buffer_size, (1,), generator=self.generator).item()
            out = self.buffer[i]
            if self.buffer_size > self.window_size:
                # If buffer is large, pop last item into the freed slot.
                self.buffer[i] = self.buffer[self.buffer_size - 1]
                self.buffer_size -= 1
            else:
                # If buffer is small, add new item into the freed slot.
                self.buffer[i] = first_draw if first_draw is not None else next(dataset)
                first_draw = None
            yield out

    def _pad_buffer(self):
        if len(self.buffer) < self.window_size:
            self.buffer += [
                [],
            ] * (self.window_size - len(self.buffer))

    def state_dict(self):
        # Create generator if it doesn't already exist
        self.setup()
        # Write generator state manually
        self.g_state = self.generator.get_state().clone().tolist()
        # Prune buffer so it can be resharded in future
        self.buffer = _int_tensor(self.buffer[: self.buffer_size])
        out = super().state_dict()
        # Pad buffer back out again
        self.buffer = self.buffer.tolist()
        self._pad_buffer()
        return out

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.buffer = self.buffer.tolist()
        # Manually set generator state if it exists
        if self.g_state is not None:
            self.generator.set_state(torch.tensor(self.g_state, dtype=torch.uint8))
        # Manually set buffer size
        self.buffer_size = len(self.buffer)


class DictShuffleDataset(_NestedStatefulDataset):
    """
    As ShuffleDataset, but for data items that are dicts of torch tensors, rather than python lists.
    Each key's tensor must have the same shape from item to item.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    window_size : int
        Target size of input/output buffer
    seed : int
        Random seed to use for shuffling
    n_data_fields : int
        Length of the dictionary comprising each data item
    """

    def __init__(self, dataset: _StatefulDataset, window_size: int, seed: int=42, n_data_fields: int=1):
        super().__init__(dataset)
        assert (
            window_size > 1
        ), f"Window size {window_size} must be greater than 1 for shuffling to occur"
        assert n_data_fields > 0, "Number of dict fields must be greater than 0"
        self.window_size = window_size
        self.g_state = None
        self.generator = None
        self.buffer: List[Dict[str, torch.tensor]] = [] 
        for i in range(n_data_fields):
            setattr(self, "buffer_"+str(i), [])
        self.data_keys: List[str] = []
        self.buffer_size = 0
        self.state_vars = ["g_state"]
        self.broadcast_vars = ["data_keys"]
        self.reshard_vars = ["buffer_"+str(i) for i in range(n_data_fields)]
        self.seed = seed
        self.n_data_fields = n_data_fields

    def setup(self):
        if not self.is_setup:
            super().setup()
            self.generator = torch.Generator().manual_seed(self.rank + self.seed)

    def __iter__(self):
        self.setup()
        dataset = iter(self.dataset)
        # Pad out buffer if needed
        self._pad_buffer()
        first_draw = next(dataset)
        # Record dict fields for state reading/writing
        self.data_keys = list(first_draw.keys())
        assert len(first_draw.keys())==self.n_data_fields, f"Num data fields ({len(first_draw.keys())}) does not match specified value ({self.n_data_fields}): {list(first_draw.keys())}"
        # If buffer entries have wrong length, reset buffer
        shape_match = True
        if len(first_draw) != len(self.buffer[0]):
            shape_match = False
        else:
            for k in first_draw.keys():
                if first_draw[k].shape != self.buffer[0][k].shape:
                    shape_match = False            
        if not shape_match:
            self.buffer = []
            self.buffer_size = 0
            self._pad_buffer()
        while True:
            # If buffer is undersized, add a datapoint
            if self.buffer_size < self.window_size:
                self.buffer[self.buffer_size] = first_draw if first_draw is not None else next(dataset)
                first_draw = None
                self.buffer_size += 1
            # Swap out randomly sampled value from buffer.
            i = torch.randint(self.buffer_size, (1,), generator=self.generator).item()
            out = self.buffer[i]
            if self.buffer_size > self.window_size:
                # If buffer is large, pop last item into the freed slot.
                self.buffer[i] = self.buffer[self.buffer_size - 1]
                self.buffer_size -= 1
            else:
                # If buffer is small, add new item into the freed slot.
                self.buffer[i] = first_draw if first_draw is not None else next(dataset)
                first_draw = None
            yield out

    def _pad_buffer(self):
        if len(self.buffer) < self.window_size:
            self.buffer += [
                {},
            ] * (self.window_size - len(self.buffer))

    def state_dict(self):
        # Create generator if it doesn't already exist
        self.setup()
        # Write generator state manually
        self.g_state = self.generator.get_state().clone().tolist()
        # Pull buffer fields into reshard vars
        buffer = self.buffer[:self.buffer_size]
        if len(self.data_keys) > 0:
            for i in range(self.n_data_fields):
                buffer_i = torch.stack([x[self.data_keys[i]] for x in buffer], dim=0)
                setattr(self, "buffer_"+str(i), buffer_i)
        out = super().state_dict()
        return out

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Pull individual buffer states into global dict buffer
        if len(self.data_keys) > 0:
            self.buffer = [{self.data_keys[j]:getattr(self, "buffer_"+str(j))[i] for j in range(self.n_data_fields)} for i in range(len(self.buffer_0))]
        # Manually set generator state if it exists
        if self.g_state is not None:
            self.generator.set_state(torch.tensor(self.g_state, dtype=torch.uint8))
        # Manually set buffer size
        self.buffer_size = len(self.buffer)


class DocPackingDataset(_NestedStatefulDataset):
    """
    Packs and slices variable-length documents into constant-length training sequences,
    attempting to minimize truncation. Maintains a list of buffers, draws a full document
    (until delimiter token is reached), and attempts to fit that document (or document remainder,
    when document is longer than target sequence length) into the fullest buffer that can contain it.
    When the number of right-padding tokens in a buffer falls below the specified threshold, that buffer
    is passed as the next sequence output. Number of buffers is set roughly to n_bins, but may rise/fall
    as documents and fragments are added/flushed. Buffers are redistributed over workers when rescaling.

    NB: currently assumes that sequences are numerical, and do not contain the value -100. This can be
    changed in future if it causes problems.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    seq_len : int
        Length of emitted training sequences
    n_pads : int
        The maximum number of right-pads allowed in a training sequence
    delimiter_token : Any
        The value that indicates the end of a document when it occurs at the end of any of the
        subdataset's emitted chunks
    pad_token : Any
        The value to use as a padding token. Data type should match the underlying data sequences
        being processed.
    n_bins : int
        The target number of buffers to maintain that are filled by incoming data chunks. Higher values
        will result in less padding, but with diminishing returns and increasing overhead.
    """
    def __init__(
            self,
            dataset: _StatefulDataset,
            seq_len: int,
            n_pads: int,
            delimiter_token: Any,
            pad_token: Any,
            n_bins: int = 100,
    ):
        super().__init__(dataset)
        self.len = seq_len
        self.delimiter = delimiter_token
        self.pad = pad_token
        self.npads = n_pads
        self.nbins = n_bins
        self.bins = []
        self.reshard_vars = ["bins"]
        self.dummy = -100
        if self.dummy == self.pad:
            self.dummy -= 1
        if self.dummy == self.delimiter:
            self.dummy -= 1

    def _available_bins(self, targ):
        # Find the bins with enough space to accomodate a chunk of target length
        slack = _int_tensor(self.bins).eq(self.dummy).flip(dims=(1,)).cumprod(dim=1).sum(dim=1)
        n_available = slack.ge(targ).int().sum().item()
        return n_available, slack

    def _bin_insert(self, slack, doc):
        # Insert given doc into the fullest bin that can accommodate it
        slack_after = slack.sub(len(doc))
        slack_after += slack_after.sign().clamp(min=-1,max=0).neg().mul(1e12).long()
        best_bin = slack_after.argmin().item()
        start = self.len - slack[best_bin].item()
        self.bins[best_bin][start:start+len(doc)] = doc

    def __iter__(self):
        self.setup()
        dataset = iter(self.dataset)
        # If seq len doesn't match current bucket size, dump current buckets
        if len(self.bins) == 0 or len(self.bins[0]) != self.len:
            self.bins = [[self.dummy]*self.len]
        while True:
            # Flush any sufficiently full buckets
            n_underfull,slack = self._available_bins(self.npads+1)
            n_yield = len(self.bins) - n_underfull
            if n_yield > 0:
                self.bins.sort(key=lambda x: _int_tensor(x).eq(self.dummy).flip(dims=(0,)).cumprod(dim=0).sum().neg())
                n_underfull,slack = self._available_bins(self.npads+1)
                for i in range(n_yield):
                    # Count forward to flush oldest buckets (of same length) first
                    out = self.bins.pop(i-n_yield)
                    # Replace any dummy tokens with pads
                    n_dummies = slack[i-n_yield].item()
                    if n_dummies > 0:
                        out[-n_dummies:] = [self.pad] * n_dummies
                    yield out
                slack = slack[:-n_yield]

            # If bin count is under target and no empty bins already exist,
            # add a single new empty bin (grow smoothly to run smoothly)
            if len(self.bins) == 0 or (slack.max() < self.len and len(self.bins) < self.nbins):
                self.bins.append([self.dummy]*self.len)

            # Fetch a doc
            doc = []
            doc_trunc = False
            while len(doc)==0 or doc[-1] != self.delimiter:
                doc += next(dataset)
            # If doc is large, add as many full buckets as needed
            while len(doc) > self.len:
                self.bins.append(doc[:self.len])
                doc = doc[self.len:]
                doc_trunc = True

            # Insert (remaining) doc into existing buckets
            if len(doc) > 0:
                # Determine if doc fits into existing buckets
                n_available,slack = self._available_bins(len(doc))
                if n_available > 0:
                    # Add doc to fullest available bin
                    self._bin_insert(slack, doc)
                else:
                    # If doc isn't truncated, or we have too many bins,
                    # truncate and fill one bin before adding another
                    if not doc_trunc or len(self.bins) >= self.nbins:
                        # Find the fullest non-full bin
                        best_bin = slack.add(slack.sign().sub(1).neg().mul(1e12).long()).argmin().item()
                        self.bins[best_bin][-slack[best_bin].item():] = doc[:slack[best_bin].item()]
                        doc = doc[slack[best_bin].item():]
                        n_available,slack = self._available_bins(len(doc))
                    # Repeat the fit-check since above code might have shortened the doc fragment
                    if n_available > 0:
                        # Add doc to fullest available bin
                        self._bin_insert(slack, doc)
                    else:
                        # Don't re-truncate, just create a new bin
                        self.bins.append(doc + [self.dummy] * (self.len - len(doc)))

    def state_dict(self):
        # Convert self.bins to tensor
        self.bins = _int_tensor(self.bins)
        out = super().state_dict()
        # Convert tensor back to nested list
        self.bins = self.bins.tolist()
        return out

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Convert tensor to nested list
        self.bins = self.bins.tolist()


class SamplingDataset(_NestedStatefulDataset):
    """
    A _NestedStatefulDataset implementing percentage-based sampling: weights can be floats, and the
    number of tokens seen from each subdataset will match those weights as closely as possible.
    This is accomplished by maintaining a _StatefulDataset for each subdataset, and tracking
    the number of tokens emitted by each. Whichever loader is furthest from its target will be
    the next to pass a document.
    ...
    Args
    ----
    datapath : str
        Absolute path to the dataset directory. Expects directory to contain subfolders,
        which in turn contain shard files. Overrides path attribute of instantiated dataset arg.
    dataset : _StatefulDataset
        Fully instantiated dataset. Cloned across desired subdatasets during setup.
    delimiter_token : Any
        The value that indicates the end of a document when it occurs at the end of any of the
        subdatasets' emitted chunks
    datasets : list[str] | None
        A list of subfolders to draw from. If None, draws from all non-nested subfolders of datapath.
        Supports relative paths in case of nested subfolders.
    weights : list(float) | None
        Weights describing what percent of emitted tokens should come from each subdataset.
        Need not sum to 1. If None, tokens are drawn evenly.
    verbose : bool
        Track setup progress?
    """

    def __init__(
        self,
        datapath: str,
        dataset: _StatefulDataset,
        delimiter_token: Any,
        datasets=None, # TODO: rename this argument to folders or paths
        weights=None,
        verbose=False,
    ):
        super().__init__(dataset)
        self.datapath = datapath
        self.delimiter = delimiter_token
        self.verbose = verbose
        self.datasets = (
            datasets
            if datasets is not None
            else [
                f
                for f in os.listdir(datapath)
                if not os.path.isfile(os.path.join(datapath, f))
            ]
        )
        assert len(self.datasets) > 0, "You must specify at least one dataset"
        if weights is not None:
            assert len(weights) == len(
                self.datasets
            ), f"Number of oversample weights {len(weights)} must match number of datasets {len(self.datasets)}"
            for w in weights:
                assert w > 0, f"Sampling rate {w} must be positive"
        self.weights = [1] * len(self.datasets) if weights is None else weights
        self.weights = [w / sum(self.weights) for w in self.weights]

        self.tokens_seen = [0] * len(self.datasets)
        self.current_iterator = -1
        self.state_vars = ["tokens_seen", "current_iterator"]

    def setup(self):
        if not self.is_setup:
            _StatefulDataset.setup(self)
            # Build subdataset iterators
            data = []
            for i, d in enumerate(self.datasets):
                data.append(deepcopy(self.dataset))
                data[-1].datapath = os.path.join(self.datapath, d)
                data[-1].rank = self.rank
                data[-1].worldsize = self.worldsize
                data[-1].local_worldsize = self.local_worldsize
                if self.verbose and self.rank == 0:
                    print(
                        f"Assembled subdataset iterator for {d}, {i+1} of {len(self.datasets)}"
                    )
            self.dataset = data
            [d.setup() for d in data]

    def __iter__(self):
        self.setup()
        # Grab one doc at a time in random order
        data = [iter(d) for d in self.dataset]
        while True:
            if self.current_iterator != -1:
                # Finish current document
                out = next(data[self.current_iterator])
                self.tokens_seen[self.current_iterator] += len(out)
                if out[-1] == self.delimiter:
                    self.current_iterator = -1
                yield out
            else:
                # Choose new subdataset to draw from (whichever is currently most underrepresented
                # compared to target ratios)
                offset = [
                    self.tokens_seen[i]
                    - self.weights[i] * sum(self.tokens_seen)
                    for i in range(len(self.datasets))
                ]
                offset_argmax = min((diff, i) for i, diff in enumerate(offset))[1]
                self.current_iterator = offset_argmax


class TitanMMPackingDataset(_NestedStatefulDataset):
    """
    TODO
    """

    def __init__(
        self,
        dataset: _StatefulDataset,
        packer: Any,  # Titan packer, fully instantiated
    ):
        super().__init__(dataset)
        self.packer = packer

        # Packer states
        self.packer_buffers_state = None
        self.packer_samples_state = None

        self.custom_vars = ["packer_buffers_state", "packer_samples_state"]
        self.custom_fns = [
            lambda x: atomic_rescale(x, self.rank, self.worldsize),
            lambda x: atomic_rescale(x, self.rank, self.worldsize),
        ]

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            out = next(dataset)
            self.packer.add_sample(out)
            if self.packer.has_batch_ready():
                batch = self.packer.get_next_batch()
                if batch:
                    yield from batch

    def state_dict(self):
        # Write packer's state into shard state. Use pickled lists to prevent DCP
        # from breaking down list-valued states into subvariables with indexed keys
        self.packer_buffers_state = pickle.dumps(list(self.packer.sample_buffer))
        self.packer_samples_state = pickle.dumps(list(self.packer.packed_samples))
        return super().state_dict()
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if not isinstance(self.packer_buffers_state, List):
            # If not rescaling, unpickle list-valued state vars
            print(f".   Rank {self.rank}: Unpickling")
            self.packer_buffers_state = pickle.loads(self.packer_buffers_state)
            self.packer_samples_state = pickle.loads(self.packer_samples_state)
            print(f".   Rank {self.rank}: Unpickled")
        else:
            # If rescaling, pickle_atomic_rescale returns a list of states. 
            # Extract/merge relevant list entries
            def list_state_handler(state):
                if len(state) == 0:
                    return []
                elif len(state) == 1:
                    return state[0]
                else:
                    return sum(state, [])
            self.packer_buffers_state = list_state_handler([pickle.loads(x) for x in self.packer_buffers_state])
            self.packer_samples_state = list_state_handler([pickle.loads(x) for x in self.packer_samples_state])
        # Read shard state into packer's state
        print(f".   Rank {self.rank}: Loading")
        self.packer.sample_buffer = deque(self.packer_buffers_state)
        self.packer.packed_samples = deque(self.packer_samples_state)
        print(f".   Rank {self.rank}: Loaded")