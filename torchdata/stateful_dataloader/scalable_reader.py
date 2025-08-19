import logging
import math
import os
import pyarrow as pa
import tempfile
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, Callable, List, Optional, Set, Tuple

import torch
# from torch.distributed import checkpoint
# from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
# import torch.distributed.tensor as dtensor
# import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data import DataLoader

from .stateful_dataloader import StatefulDataLoader

"""
This file borrows the StatefulDataset framework from the IBM fms-fsdp repo to implement rescalable data
loading. This framework is analogous to the existing torchdata nodes framework and will be converted
in the future.

Rescalability is implemented at the base level - you must use this layer to interface with a collection
of indexable files directly. The ScalableReader then yields data values like an iterator. These values
are not shuffled. 

ScalableReader interfaces with indexable files via custom FileHandlers. These FileHandlers implement basic
file operations such as file type checking, opening, indexing, and slicing. By implementing these basic
operations, users can add support for arbitrary file types.

Rescalability is implemented by splitting data into a large number of logical shards, which are then
allocated over the set of dataloader workers. We assume that logical shards vastly outnumber workers,
such that when workers do not divide logical shards evenly, the off-by-one allocations don't matter and
workers still finish their epochs at roughly the same time. Files are assigned to logical shards
fractionally and based on file size, such that each shard contains roughly equal amounts of data, and as
few individual files as possible. This minimizes the number of file pulls. 

ScalableReaders step through a single active logical shard at a time, to minimize overhead. This behavior
can be relaxed later.

When rescaling to a different number of workers, the logical shard progress counters are aggregated
globally onto each ScalableReader. Then, completed and incomplete logical shards are re-allocated
separately, to ensure that each worker receives roughly the same ratio of seen to unseen data in the
current epoch. This allows us to scale from any number of workers to any other number.

State dict saving and loading behavior is governed by tagging the relevant class variables as one of 4
options: 1) state (scalar values dropped when rescaling), 2) broadcast (saved values identical across all 
workers), 3) reshard (tensors that are repartitioned on dim 0 when rescaling), and 4) custom (paired with 
a user-provided resharding function, for when more sophisticated behavior is required). The base 
_StatefulDataset stub illustrates usage. 

Differently tagged state values are saved under separate state sub-dictionaries. A separate saving/loading
framework is required for aggregating state dicts from workers and saving/loading to disk. The current
code is a custom naive implementation, and will be deprecated to testing purposes as we pivot
to DCP integration.
"""


#### -------------------------    BORROWED FROM IBM FMS-FSDP    ------------------------- ####

class _StatefulDataset(data.IterableDataset):
    """
    Stub for stateful datasets, extends data.IterableDataset with state_dict methods.
    All subclasses should specify the variables to be considered stateful via the provided tag lists.
    State, Broadcast, Reshard, and Custom state variables should be assigned to the relevant list
    (e.g. self.state_vars, self.broadcast_vars, etc.)
    """

    def __init__(
        self,
        datapath: str,
        rank: int,
        worldsize: int,
    ):
        assert rank >= 0, f"Rank {rank} must be a positive integer"
        assert worldsize > rank, f"Worldsize {worldsize} must be greater than rank {rank}"
        assert datapath is None or (
            os.path.isdir(datapath) and len(os.listdir(datapath)) > 0
        ), f"Data path {datapath} must be a non-empty folder or None"

        # Default fields
        self.datapath = datapath
        self.rank = rank
        self.worldsize = worldsize
        self.local_worldsize = -1

        # Setup / loading flags
        self.is_setup = False

        # Tag lists for state saving/loading
        self.state_vars: List[str] = []
        self.broadcast_vars: List[str] = []
        self.reshard_vars: List[str] = []
        self.custom_vars: List[str] = []
        self.custom_fns: List[Callable[[List[Any]], Any]] = []
        # Every custom var must be bundled with a corresponding custom resharding fn,
        # which maps the list of prior values over workers to the new value for this worker. 
        # It is recommended to define these fns as class methods, so that rank and worldsize are exposed.

    def setup(self):
        """
        This method should contain all setup depending on datapath or rank.
        It is called after init, but immediately before any other operation.
        Certain operations higher up in the pipeline may change rank or datapath
        after init (for example, wrapping in a subdataset sampler layer, or copying
        to worker processes), so all rank- and datapth- dependent ops are deferred to
        this function.
        Currently, this function simply adjusts rank/worldsize to account for
        multiprocess dataloaders.
        """
        if not self.is_setup:
            self.is_setup = True
            # Perform adjustment only if not already adjusted (i.e. via _WrapperDataset)
            if self.local_worldsize == -1:
                info = data.get_worker_info()
                if info is None or info.num_workers == 1:
                    # No multi-worker rank adjustment needed
                    self.local_worldsize = 1
                else:
                    self.local_worldsize = info.num_workers
                    self.worldsize = self.worldsize * self.local_worldsize
                    self.rank = self.local_worldsize * self.rank + info.id

    def statename(self, x: str, rank=None):
        # Note that this naming convention implicitly disallows repeated layers in the dataset pipeline
        out = self.__class__.__name__ + "."
        if rank is not None:
            out += str(rank) + "."
        out += x
        return out

    def state_dict(self):
        """
        Retrieve all state vars (each worker/process produces its own state dict shard).
        On the off chance that you're saving a checkpoint with zero steps, run setup first.
        """
        self.setup()
        out = {}
        for state_type,flags in zip(
            ["state", "broadcast", "reshard", "custom"],
            [self.state_vars, self.broadcast_vars, self.reshard_vars, self.custom_vars],
        ):
            out[state_type] = {self.statename(flag): getattr(self, flag) for flag in flags}
        # Deepcopy required to prevent in-place modification from later prefetches
        return deepcopy(out)

    def load_state_dict(self, state_dict):
        """
        Run setup if needed, and apply all applicable state vars from the state_dict.
        """
        self.setup()
        for state_type,flags in zip(
            ["state", "broadcast", "reshard", "custom"],
            [self.state_vars, self.broadcast_vars, self.reshard_vars, self.custom_vars],
        ):
            [setattr(self, flag, state_dict[state_type][self.statename(flag)]) for flag in flags]
        # Apply custom reshard fns to loaded custom values
        [setattr(
            self, 
            self.custom_vars[i], 
            self.custom_fns[i](getattr(self, self.custom_vars[i]))
        ) for i in range(len(self.custom_vars))]


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
                subdict = {state_type:{k[len(prefix):]:v} 
                           for state_type in state_dict 
                           for k,v in state_dict[state_type].items()
                           if prefix in k}
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
            out = self.dataset.state_dict()
            for state_type in state.keys():
                out[state_type].update(state[state_type])
        else:
            for i,subdata in enumerate(self.dataset):
                substate = subdata.state_dict()
                for state_type in state.keys():
                    out[state_type].update(
                        {self.statename(k,i):v for k,v in substate[state_type].items()}
                    )
        return out


#### -------------------------    FILE HANDLERS    ------------------------- ####


class ShardFileHandler(object, metaclass=ABCMeta):
    """
    Stub for shard file readers of different formats.
    Must implement open, length, indexing, and slicing functions.
    """

    def is_legal(self, filepath: str):
        """
        Given a file path, determine if it qualifies for this handler.
        Ideally does not involve opening the file.
        """
        return os.path.isfile(filepath)

    @abstractmethod
    def open(self, path: str):
        """
        Open the file, to be indexed via self.get() method.
        Avoid reading entire multi-Gb files when possible!
        """
        pass

    @abstractmethod
    def length(self, path: str):
        """
        Calculate the number of documents in the given file.
        Avoid reading entire multi-Gb files when possible!
        """
        pass

    @abstractmethod
    def get(self, reader, index: int, drop_tokens: Set):
        """
        Given the output of self.open() and an index, return the document at that index.
        Then, remove the first and/or last items if they appear in drop_tokens.
        Try to avoid reading entire documents at a time in case of long documents,
        but this is less important than avoiding reading entire files as above.
        Output must support len() method.
        """
        pass

    @abstractmethod
    def slice(self, doc, index: int, n_pull: int) -> List:
        """
        Given a long document, retrieve n_pull consecutive items starting from index.
        Again, try to be memory-efficient when doing so, but efficiency in self.get()
        and self.open() is far more important.
        Must return a python list.
        """
        pass


class ArrowHandler(ShardFileHandler):
    """
    Reader for indexable, pre-tokenized PyArrow shard files.
    Pyarrow shard files are expected to hold multiple RecordBatches,
    where each RecordBatch has a "tokens" field consisting of
    a single token list (i.e. each document is a single sequence
    under a "token" field, and the file is a list of such sequences).

    A preferred format as we can load document chunks without having to ever pull
    the entire document or shard file, allowing for graceful handling of large documents.
    Non-standard data format, though.
    """

    def __init__(self, col_name: str = "tokens"):
        self.col_name = col_name

    def is_legal(self, filepath: str):
        return "arrow" in os.path.splitext(filepath)[1]

    def open(self, path: str):
        return pa.ipc.open_file(pa.memory_map(path))

    def length(self, path: str):
        return self.open(path).num_record_batches

    def get(self, reader: pa.RecordBatchFileReader, index: int, drop_tokens: Set):
        doc = reader.get_batch(index)[self.col_name]
        if len(doc) > 0 and doc[0].as_py() in drop_tokens:
            doc = doc.slice(1, len(doc) - 1)
        # Recheck len for edge case where doc=[eos]
        if len(doc) > 0 and doc[-1].as_py() in drop_tokens:
            doc = doc.slice(0, len(doc) - 1)
        return doc

    def slice(self, doc: pa.UInt32Array, index: int, n_pull: int) -> List:
        return doc.slice(index, n_pull).to_pylist()
        
        
#### -------------------------    DATASET LAYERS    ------------------------- ####


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


#### -------------------------    NEW CODE STARTS HERE    ------------------------- ####


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
        while True:
            # If buffer entries have wrong length, reset buffer
            if len(first_draw) != len(self.buffer[0]):
                self.buffer = []
                self.buffer_size = 0
                self._pad_buffer()

            # If buffer is undersized, add a datapoint
            if self.buffer_size < self.window_size:
                self.buffer[self.buffer_size] = next(dataset) if self.buffer_size > 0 else first_draw
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
                self.buffer[i] = next(dataset)
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
        self.g_state = self.generator.get_state()
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
            self.generator.set_state(self.g_state)
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
            if slack.max() < self.len and len(self.bins) < self.nbins:
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
    
    
class ScalableReader(_StatefulDataset):
    """
    Maintains n x 5 state buffer where n is the number of logical shards owned by this worker,
    and 5 is the number of relevant data fields per-shard. Finishes shards with the lowest
    visit count before continuing into new epoch. When rescaling, re-allocates visited / unvisited
    shards in the current epoch separately, so that each new worker finishes the epoch at around
    the same time.

    Currently does not shuffle docs within shards/files, but this can be added later.
    """

    def __init__(
        self, 
        datapath: str, 
        rank: int, 
        worldsize: int,
        filehandler: ShardFileHandler,
        delimiter_token: Any,
        bos_token: Optional[Any] = None,
        strip_tokens: Optional[Set[Any]] = set(),
        seed: int = 42,
        min_length: int = 1,
        max_chunksize: int = 1024,
        n_logical_shards: int = 30720,
        verbose: bool = False,
    ):
        super().__init__(datapath, rank, worldsize)
        self.seed = seed  # Currently unused
        self.datapath = datapath
        self.filehandler = filehandler()
        self.min_length = min_length  # Ignore any docs shorter than this
        assert max_chunksize > 0, f"Max chunksize must be a nonzero positive integer"
        self.chunksize = max_chunksize  # Yield chunks at a time if doc is longer than this
        self.eos = delimiter_token  # Inserted between each doc
        self.bos = bos_token  # Inserted before each doc (optional)
        self.drop = strip_tokens  # Tokens to drop from begin/end of doc (replaced by above delimiter/bos)
        self.n_logical_shards = n_logical_shards
        self.verbose = verbose  # Currently unused
        
        # Position
        self.reader = None
        self.cur_file = None

        # Setup flags
        self.is_setup = False
        self.filesizes = None  # [[filenames], [filesizes]]  (constructed pre-iter if not loaded from ckp)
        self.shard_states = None  # shardid, file pos, doc pos, chunk pos, epoch   (reshardable state buffer)

        self.broadcast_vars = ["filesizes"]
        self.custom_vars = ["shard_states"]
        self.custom_fns = [self.shard_rescale]

        # TODO: add handling to prevent zero-length allocations

    def _get_shard_breakdown(self, rank, nshards):
        """
        Retrieve the set of (fractional) files assigned to a given logical shard
        """
        # Find first doc included in the current shard
        sizelist = torch.tensor(self.filesizes[1])
        sizelist = sizelist/sizelist.float().sum()
        cum_sizelist = sizelist.cumsum(0)
        start_frac = rank/nshards
        start_id = len(sizelist) - cum_sizelist.gt(start_frac).sum().item()
        # For each doc, assign relevant fractional ownership
        start = start_frac
        end = (rank+1)/nshards
        my_files = []  # fileid, start%, end%
        for i, (size, cumsize_incl) in enumerate(
            zip(sizelist[start_id:].tolist(), cum_sizelist[start_id:].tolist())
        ):
            id = start_id + i
            cumsize = cumsize_incl - size
            if cumsize > end:
                # No more files to include, stop early
                break
            elif cumsize <= end and cumsize_incl >= start:
                my_files.append([
                    id,
                    min(max((start - cumsize) / size, 0), 1),
                    min(max((end - cumsize) / size, 0), 1),
                ])
        return my_files

    def setup(self):
        """
        Perform any rank- and path-dependent setup. This operation is deferred from __init__ 
        to support multiple workers in the dataloader.
        """
        if not self.is_setup:
            # Get your adjusted rank and worldsize
            super().setup()

            # Get logical shard partitions
            my_shards = list(range(
                (self.n_logical_shards * self.rank) // self.worldsize,
                (self.n_logical_shards * (self.rank + 1)) // self.worldsize,
            ))

            # Set up logical shard states (may be overwritten later by ckp load)
            self.shard_states = torch.zeros(math.ceil(self.n_logical_shards / self.worldsize), 5, dtype=torch.int)
            self.shard_states[:len(my_shards), 0] = torch.tensor(my_shards)

            # Pad shard state if this worker is off by one. Id is -1 and visit count is inf.
            self.shard_states[len(my_shards):, 0] = -1
            self.shard_states[len(my_shards):, 4] = torch.iinfo(torch.int).max

    def _pre_iter(self):
        """
        Construct index of data files and their filesizes. 
        This is saved/loaded in subsequent checkpoints to avoid re-indexing the entire dataset repeatedly.
        """
        # Assemble set of available shard files, if nonexistant
        if self.filesizes is None:
            # Find all legal files
            shards = [
                [os.path.join(root,name)[len(self.datapath)+1:], os.path.getsize(os.path.join(root, name))]
                for root, dirs, files in os.walk(self.datapath, topdown=False)
                for name in files
                if self.filehandler.is_legal(os.path.join(root, name))
            ]
            shards.sort()
            # Flip list of (shard,size) tuples into (shardlist,sizelist)
            self.filesizes = list(zip(*shards)) 

    def _get_reader(self, fileid, reader, ndocs):
        """
        If new fileid does not match the current one, open a new reader on
        the corresponding filepath. Also return the number of docs in the file.
        """
        if self.cur_file == fileid:
            return reader, ndocs
        else:
            self.cur_file = fileid
            filepath = os.path.join(self.datapath, self.filesizes[0][fileid])
            return self.filehandler.open(filepath), self.filehandler.length(filepath)

    def _construct_chunk(self, j, doc, n_chunks):
        """
        Grab a chunk of the desired size from the document, with eos/bos handling
        """
        start_index = j * self.chunksize
        n_pull = self.chunksize
        if self.bos is not None:
            if j == 0:
                n_pull -= 1
            else:
                start_index -= 1
        chunk = self.filehandler.slice(doc, start_index, n_pull)
        # Add bos/eos tokens if needed
        if self.bos is not None and j == 0:
            chunk = [self.bos] + chunk
        if j == n_chunks - 1:
            chunk = chunk + [self.eos]
        return chunk
    
    def __iter__(self):
        if not self.is_setup:
            self.setup()
        self._pre_iter()
        reader = None
        ndocs = -1
        while True:
            # Isolate undervisited shards
            epoch_count = self.shard_states[:,4].min().item()
            shardset = self.shard_states[:,4].eq(epoch_count).nonzero().squeeze(-1)
            for i in shardset:
                shardid = self.shard_states[i][0].item()
                files = self._get_shard_breakdown(shardid, self.n_logical_shards)  # list([docid, start%, end%])
                file_offset = self.shard_states[i][1].item()
                for file_pos in range(file_offset, len(files)):
                    # Update position
                    self.shard_states[i][1] = file_pos
                    # Calculate doc range
                    file = files[file_pos]
                    fileid = file[0]
                    reader, ndocs = self._get_reader(fileid, reader, ndocs)
                    doc_start = round(ndocs * file[1])
                    doc_end = round(ndocs * file[2])
                    doc_offset = self.shard_states[i][2].item()
                    for doc_pos in range(doc_offset, doc_end - doc_start):
                        # Update position
                        self.shard_states[i][2] = doc_pos
                        # Fetch doc
                        doc = self.filehandler.get(reader, doc_start + doc_pos, self.drop)
                        doclen = len(doc)
                        nchunks = math.ceil(doclen/self.chunksize)
                        chunk_offset = self.shard_states[i][3].item()
                        for chunk_pos in range(chunk_offset, nchunks):
                            # Update position
                            self.shard_states[i][3] = chunk_pos+1
                            # Yield chunk
                            yield self._construct_chunk(chunk_pos, doc, nchunks)
                        # Reset chunk_pos after finishing doc
                        self.shard_states[i][3] = 0
                    # Reset doc_pos after finishing file
                    self.shard_states[i][2] = 0
                # Reset file_pos after finishing shard
                self.shard_states[i][1] = 0
                # Increase epoch count after finishing shard
                self.shard_states[i][4] += 1
            # Begin new epoch
    
    def shard_rescale(self, shard_states: List[torch.Tensor]):
        """
        Custom function for rescaling of ScalableReader.shard_states
        """
        if len(shard_states) == self.worldsize:
            return shard_states[self.rank]
        else:
            # Sort shards by epoch count
            shard_states = torch.cat(shard_states, dim=0)
            sorted, indices = torch.sort(shard_states[:,4], descending=True, stable=True)
            shard_states = shard_states[indices]
            # Strip out dummy padding shards
            n_dummies = sorted.eq(torch.iinfo(torch.int).max).sum()
            shard_states = shard_states[n_dummies:]  # n_logical 5
            sorted = sorted[n_dummies:]
            # Split into max and non-max epochs
            n_complete = sorted.eq(sorted[0]).sum()
            completed_shards = shard_states[:n_complete]
            incomplete_shards = shard_states[n_complete:]
            # Allocate completed shards
            completed_shards = [
                completed_shards[
                    round(i*len(completed_shards)/self.worldsize):
                    round((i+1)*len(completed_shards)/self.worldsize)
                ] for i in range(self.worldsize)
            ]
            # Sort completed shards by length
            completed_shards.sort(key=len)
            # Allocate incomplete shards
            incomplete_shards = [
                incomplete_shards[
                    round(i*len(incomplete_shards)/self.worldsize):
                    round((i+1)*len(incomplete_shards)/self.worldsize)
                ] for i in range(self.worldsize)
            ]
            # Reverse sort incomplete shards by length
            # Minimizes padding by overallocating incomplete shards to underallocated complete shards
            incomplete_shards.sort(key=len, reverse=True)
            
            # Pull out shard allocation for this worker
            # (sort/reverse-sort ensures allocations are off by no more than 1)
            shards = [
                completed_shards[self.rank],
                incomplete_shards[self.rank]
            ]
            shard_states = torch.cat(shards)
            # Order shards by global ID (for steady file progression)
            _, indices = shard_states[:,0].sort()
            shard_states[:len(shard_states)] = shard_states[indices]
            # Pad out with dummy shards if needed
            shard_states[len(shard_states):,0] = -1
            shard_states[len(shard_states):,4] = torch.iinfo(torch.int).max
            return shard_states


#### -------------------------    CHECKPOINT FUNCTIONS    ------------------------- ####


def save_ckpt_custom(
    loader: StatefulDataLoader,
    path: str,
):
    """
    Retrieves dataloader state dict, and separates worker states from loader state.
    Aggregates worker states, and separates out state/broadcast/reshard/custom variables.
    Saves each state dict separately. 
    """
    rank = loader.dataset.rank
    state = deepcopy(loader.state_dict())
    dstate = state["_snapshot"]["_worker_snapshots"]
    dstate = [dstate[f"worker_{i}"].pop("dataset_state") for i in range(len(dstate))]  # List[dict]
    # Flip List[dict[dict]] to dict[List[dict]]
    dstate = {k:[d[k] for d in dstate] for k in dstate[0].keys()}  # {state, broadcast, reshard, custom}

    # State dict: add loader state and save as is
    state_vars = dstate["state"]
    state_vars.append(state)
    torch.save(
        state_vars,
        os.path.join(path, f"loader_state_{rank}.pth"),
    )
    
    # Broadcast dict: Save only first entry, only if rank is 0
    if rank == 0:
        broadcast_vars = dstate["broadcast"][0]
        torch.save(
            broadcast_vars,
            os.path.join(path, f"loader_broadcast.pth"),
        )
    
    # Reshard dict: Assert is tensor and aggregate values
    reshard_vars = dstate["reshard"]
    # Assert all reshard vals are tensors
    for k,v in reshard_vars[0].items():
        assert isinstance(v, torch.Tensor), f"Reshard var {k} is not a torch tensor!"
    # Flip list[dict] to dict[list]
    reshard_vars = {k:[d[k] for d in reshard_vars] for k in reshard_vars[0].keys()}
    torch.save(
        reshard_vars,
        os.path.join(path, f"loader_reshard_{rank}.pth"),
    )
    
    # Custom dict: Aggregate lists of (val,fn) tuples into tuples of (list[val], fn)
    # (assumes fn is consistent across ranks)
    custom_vars = dstate["custom"]
    # Flip list[dict] into dict[list]
    custom_vars = {k:[d[k] for d in custom_vars] for k in custom_vars[0].keys()}
    torch.save(
        custom_vars,
        os.path.join(path, f"loader_custom_{rank}.pth"),
    )


def load_ckpt_custom(
    loader: StatefulDataLoader,
    path: str,
):
    """
    Retrieves dataloader state dict, and separates worker states from loader state.
    Handle loading/rescaling for the 4 tags.
    """
    base = loader.state_dict()
    nworkers = base["_snapshot"]["_main_snapshot"]["_num_workers"]
    r = loader.dataset.rank
    w = loader.dataset.worldsize
    dstate = base["_snapshot"]["_worker_snapshots"]
    dstate = [dstate[f"worker_{i}"].pop("dataset_state") for i in range(len(dstate))]  # List[dict]
    # Flip List[dict[dict]] to dict[List[dict]]
    dstate = {k:[d[k] for d in dstate] for k in dstate[0].keys()}  # {state, broadcast, reshard, custom}    inp = {"state":deepcopy(base), "dstate":dstate}
    
    ckp_ws = 0 if not os.path.exists(path) else len([x for x in os.listdir(path) if "loader_state_" in x])
    easy_load = False
    if ckp_ws == w:
        state_vars = torch.load(os.path.join(path, f"loader_state_{r}.pth"))
        loader_state = state_vars.pop(-1)
        easy_load = nworkers == loader_state["_snapshot"]["_main_snapshot"]["_num_workers"]
    
    # State: load if easy, otherwise ignore
    if easy_load:
        # Loader state
        base = loader_state
        # Worker states
        dstate["state"] = state_vars

    # Broadcast: load across all cases
    broadcast_vars = torch.load(os.path.join(path, f"loader_broadcast.pth"))
    dstate["broadcast"] = [broadcast_vars] * nworkers

    # Reshard: if easy, flip labels; else concat and reshard
    if easy_load:
        reshard_vars = torch.load(os.path.join(path, f"loader_reshard_{r}.pth"))
        # Flip dict[list] back to list[dict]
        reshard_vars = [{k:reshard_vars[i][k] for k in reshard_vars} for i in range(nworkers)]
        dstate["reshard"] = reshard_vars
    else:
        # Load all shards
        reshard_vars = [torch.load(os.path.join(path, f"loader_reshard_{i}.pth")) for i in range(ckp_ws)]  # list[dict[list]]
        # Flip list[dict[list]] to dict[list[list]]
        reshard_vars = {k:[reshard_vars[i][k] for i in range(ckp_ws)] for k in reshard_vars[0]}
        # Conjoin and concat inner lists: dict[list[list]] -> dict[tensor]
        reshard_vars = {k:torch.cat(sum(reshard_vars[k], []), dim=0) for k in reshard_vars}
        # For each local worker, pull out relevant shard
        reshard_state = [{} for _ in range(nworkers)]
        for local_r in range(r*nworkers, r*nworkers+nworkers):
            for k in reshard_vars:
                val = reshard_vars[k]
                reshard_state[local_r-r*nworkers][k] = val[
                    round(val.size(0)*local_r/(w*nworkers)) : round(val.size(0)*(local_r+1)/(w*nworkers))
                ]
        dstate["reshard"] = reshard_state

    # Custom: if easy, flip labels; else concat and run custom fn
    if easy_load:
        custom_vars = torch.load(os.path.join(path, f"loader_custom_{r}.pth"))
        # Flip dict[list] into list[dict]
        custom_vars = [{k:custom_vars[k][i] for k in custom_vars} for i in range(nworkers)]
        dstate["custom"] = custom_vars
    else:
        # Load all shards
        custom_vars = [torch.load(os.path.join(path, f"loader_custom_{i}.pth")) for i in range(ckp_ws)]  # list[dict[tuple[list]]]
        # Flip and fuse list[dict[list]] into dict[list]
        custom_vars = {k:sum([c[k] for c in custom_vars], []) for k in custom_vars[0]}
        # Expand dict[list] to list[dict[list]] via replication
        dstate["custom"] = [custom_vars] * nworkers

    # Flip dict[list[dict]] into list[dict[dict]]
    dstate = [{k:dstate[k][i] for k in dstate} for i in range(nworkers)]
    # Load worker dstates back into loader
    for i in range(nworkers):
        base["_snapshot"]["_worker_snapshots"][f"worker_{i}"]["dataset_state"] = dstate[i]
    loader.load_state_dict(base)


def dummydata():
    data = tempfile.TemporaryDirectory()
    datapath = data.name
    schema = pa.schema([pa.field("tokens", pa.uint32())])
    with pa.ipc.new_file(
        os.path.join(datapath, "fileshard_1.arrow"), schema
    ) as writer:
        for i in range(500):
            out = list(range(i * 100, i * 100 + 100))
            writer.write(pa.record_batch([out], schema=schema))
    os.makedirs(os.path.join(datapath, "subfolder"))
    with pa.ipc.new_file(
        os.path.join(datapath, "subfolder/fileshard_2.arrow"), schema
    ) as writer:
        for i in range(500):
            out = list(range(50000 + i * 100, 50000 + i * 100 + 100))
            writer.write(pa.record_batch([out], schema=schema))
    return data

def dummytest():
    data = dummydata()
    path=data.name
    test = ScalableReader(path, 0, 1, ArrowHandler, -1, n_logical_shards=10)
    l = StatefulDataLoader(test, batch_size=1, num_workers=2)
    for i,out in enumerate(l):
        if i==463:
            break
    save_ckpt_custom(l, path)
    print(l.state_dict())

    test2 = ScalableReader(path, 2, 5, ArrowHandler, -1, n_logical_shards=10)
    l2 = StatefulDataLoader(test2, batch_size=1, num_workers=2)
    load_ckpt_custom(l2, path)
    out = iter(l2)
    print(next(out)[0])
    print(l2.state_dict())

def docpacktest():
    data = dummydata()
    path=data.name
    test = ScalableReader(path, 0, 1, ArrowHandler, -1, n_logical_shards=10)
    test = DocPackingDataset(test, 30, 0, -1, -2, 4)
    l = StatefulDataLoader(test, batch_size=1, num_workers=2)
    for i,out in enumerate(l):
        if i==480:
            break
    save_ckpt_custom(l, path)
    print(l.state_dict())

    test2 = ScalableReader(path, 0, 5, ArrowHandler, -1, n_logical_shards=10)
    test2 = DocPackingDataset(test2, 30, 8, -1, -2, 2)
    l2 = StatefulDataLoader(test2, batch_size=1, num_workers=2)
    load_ckpt_custom(l2, path)
    out = iter(l2)
    print(next(out))
    print(next(out))
    print(next(out))
    print(l2.state_dict())

def shuffletest():
    data = dummydata()
    path=data.name
    test = ScalableReader(path, 0, 1, ArrowHandler, -1, n_logical_shards=10)
    test = ShuffleDataset(test, 4)
    l = StatefulDataLoader(test, batch_size=1, num_workers=1)
    for i,out in enumerate(l):
        if i==480:
            break
    # return
    save_ckpt_custom(l, path)
    print(l.state_dict())

    test2 = ScalableReader(path, 3, 5, ArrowHandler, -1, n_logical_shards=10)
    test2 = ShuffleDataset(test2, 4)
    l2 = StatefulDataLoader(test2, batch_size=1, num_workers=2)
    load_ckpt_custom(l2, path)
    out = iter(l2)
    print(next(out))
    print(next(out))
    print(next(out))
    print(l2.state_dict())