import os
from copy import deepcopy
from typing import Any, Callable, List

import torch

from .base_loader import _StatefulDataset


"""
TODO: blurb
"""


class _NestedStatefulDataset(_StatefulDataset):
    """
    Stub for nested wrappers of _StatefulDatasets. Extends state fns with recursion.
    Requires a single instantiated sub-dataset (which may be replicated during setup fn).
    The resulting self.dataset must either be a _StatefulDataset, or iterable of _StatefulDatasets.
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
        If multiple subdatasets are present, uses corresponding key prefixes.
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
                # for state_type in state_dict:
                #     for k,v in state_dict[state_type].items():
                #         if prefix in k:
                #             print(".   ", k[len(prefix):])
                # print(prefix, subdict, state_dict)
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


class ShuffleDataset(_NestedStatefulDataset):
    """
    Wrapper for a StatefulDataset that implements data shuffling via a single in/out buffer.
    Fills buffer two at a time, up to desired size, then switches to one at a time to maintain size.
    Passes randomly sampled outputs one by one.
    Ensures local mixing of data without relying on sliding windows or shuffling of large buffers.
    Any two consecutive inputs will be separated by window_size steps in expectation.
    Rescaling-enabled: buffers that shrink will re-grow to window_size,
    buffers that expand will shrink back down to window_size.
    Sequences retrieved from the wrapped StatefulDataset must all be the same length.
    ...
    Args
    ----
    dataset : _StatefulDataset
        Fully instantiated dataset
    window_size : int
        Max size of input/output buffer
    """

    def __init__(self, dataset: _StatefulDataset, window_size: int):
        super().__init__(dataset)
        assert (
            window_size > 1
        ), f"Window size {window_size} must be greater than 1 for shuffling to occur"
        self.window_size = window_size
        self.g_state = None
        self.generator = None
        self.buffer: List[List[Any]] = []
        self.buffer_size = 0
        self.state_vars = ["g_state"]
        self.reshard_vars = ["buffer"]

    def setup(self):
        if not self.is_setup:
            self.generator = torch.Generator().manual_seed(self.rank)
        super().setup()

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
            # If buffer is small, add new item.
            # If buffer is large, pop last item into that slot.
            i = torch.randint(self.buffer_size, (1,), generator=self.generator).item()
            out = self.buffer[i]
            if self.buffer_size > self.window_size:
                self.buffer[i] = self.buffer[self.buffer_size - 1]
                self.buffer_size -= 1
            else:
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
        self.buffer = torch.tensor(self.buffer[: self.buffer_size])
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
        

class DocPackingDataset(_NestedStatefulDataset):
    """
    Packs and slices variable-length documents into constant-length training sequences,
    attempting to minimize truncation. Maintains a list of buffers, draws a full document
    (until delimiter token is reached), and attempts to fit that document (or document remainder,
    when document is longer than target sequence length) into the fullest buffer that can contain it.
    When the number of right-padding tokens in a buffer falls below the specified threshold, that buffer
    is passed as the next sequence output. Number of buffers is set roughly to n_bins, but may rise/fall
    as documents and fragments are added/flushed. Buffers are redistributed over workers when rescaling.
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
        slack = torch.tensor(self.bins).eq(self.dummy).flip(dims=(1,)).cumprod(dim=1).sum(dim=1)
        n_available = slack.ge(targ).int().sum().item()
        return n_available, slack
    
    def _bin_insert(self, slack, doc):
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
                self.bins.sort(key=lambda x: torch.tensor(x).eq(self.dummy).flip(dims=(0,)).cumprod(dim=0).sum().neg())
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
        self.bins = torch.tensor(self.bins)
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
        which in turn contain shard files.
    dataset : _StatefulDataset
        Fully instantiated dataset. Cloned across desired subdatasets during setup.
    delimiter_token : Any
        Token used to indicate sequence/document breaks. Type should match data type.
    datasets : list[str] | None
        A list of subdatasets to draw from. If None, draws from all subfolders of datapath.
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
        datasets=None,
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
        for d in datasets:
            assert os.path.exists(
                os.path.join(datapath, d)
            ), f"Invalid subdataset path: {os.path.join(datapath, d)}"

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
                # Choose new subdataset to draw from
                # (whichever is currently most underrepresented compared to target rate)
                offset = [
                    self.tokens_seen[i]
                    - self.weights[i] * sum(self.tokens_seen)
                    for i in range(len(self.datasets))
                ]
                offset_argmax = min((diff, i) for i, diff in enumerate(offset))[1]
                self.current_iterator = offset_argmax