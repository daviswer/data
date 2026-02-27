import math
import os
from copy import deepcopy
from functools import partial
from typing import Any, Callable, List, Optional, Set

import torch
import torch.utils.data as data

from .file_handlers import ShardFileHandler
from .shard_rescaler import shard_rescale
from .shard_state import HFShardField, ShardField, ShardStateManager, TitanMMShardField

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from tokenizers import Tokenizer

"""
Implements rescalable dataloading, via a base class stub and a shard-based implementation for
interfacing with a collection of indexable data files. The ScalableReader yields data values
like an iterator, and does not perform shuffling. ScalableReader interfaces with indexable files
via custom FileHandlers from file_handlers.py.

Rescalability is implemented by splitting data into a large number of logical shards, which are then
allocated over the set of dataloader workers. We assume that logical shards vastly outnumber workers,
such that when workers do not divide logical shards evenly, the off-by-one allocations don't matter and
workers still finish their epochs at roughly the same time. Files are assigned to logical shards
fractionally and based on file size, such that each shard contains roughly equal amounts of data, and
as few individual files as possible. This minimizes the number of file pulls.

ScalableReaders step through a single active logical shard at a time, to minimize overhead.
This behavior can be relaxed later.

When rescaling to a different number of workers, the logical shard progress counters are aggregated
globally onto each ScalableReader. Then, completed and incomplete logical shards are re-allocated
separately, to ensure that each worker receives roughly the same ratio of seen to unseen data in the
current epoch. This allows us to scale from any number of workers to any other number.

State dict saving and loading behavior is governed by tagging the relevant class variables as one of 4
options: 1) state (scalar values dropped when rescaling), 2) broadcast (saved values identical across
all workers), 3) reshard (tensors that are repartitioned on dim 0 when rescaling), and 4) custom
(paired with a user-provided resharding function, for when more sophisticated behavior is required).
The base _StatefulDataset stub illustrates usage.

Differently tagged state values are saved under separate state sub-dictionaries. A separate
saving/loading framework is required for aggregating state dicts from workers and saving/loading to
disk. We leverage PyTorch Distributed Checkpointing (DCP) to implement saving, loading, and rescaling
distributed checkpoints in dcp_utils.py. A simplified, asynchronous (but also much less efficient)
implementation is provided in the unit testing script for validation and illustration purposes.
"""

# TODO: implement min_length

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
        out["custom"]["__rescaling__"] = False
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
        if state_dict["custom"]["__rescaling__"]:
            [setattr(
                self,
                self.custom_vars[i],
                self.custom_fns[i](getattr(self, self.custom_vars[i]))
            ) for i in range(len(self.custom_vars))]

    def __iter__(self):
        raise NotImplementedError
    

class ScalableTitanMMReader(_StatefulDataset):
    """
    TODO
    """

    def __init__(
        self,
        data_constructor: Any,  # Titan HuggingFaceMultiModalDataset defined up to rank and worldsize
        rank: int,
        worldsize: int,
        n_logical_shards: int = 30720,
        seed: int = 42,
    ):
        super().__init__("Dummy", rank, worldsize)
        self.data_constructor = partial(data_constructor, infinite=False)
        self.n_logical_shards = n_logical_shards
        self.seed = seed

        # Position
        self.current_shard = -1
        self.current_stream = None

        # Shard state manager (initialized in setup)
        self._shard_manager: Optional[ShardStateManager] = None

        # Packer states
        self.packer_buffers = {}
        self.packer_samples = {}

        self.custom_vars = ["shard_states", "packer_buffers", "packer_samples"]
        self.custom_fns = [
            lambda shard_states: shard_rescale(shard_states, self.rank, self.worldsize),
            self.extract_by_shard_states,
            self.extract_by_shard_states,
        ]

    def extract_by_shard_states(self, states):
        shard_inds = self._shard_manager.state[:, TitanMMShardField.SHARD_ID]
        if len(states) == self.worldsize:
            # If not rescaling, just pull out the prior state for this worker
            return states[self.rank]
        else:
            raise NotImplementedError

    def setup(self):
        """
        Perform any rank- and path-dependent setup. This operation is deferred from __init__
        to support multiple workers in the dataloader.
        """
        if not self.is_setup:
            # Get your adjusted rank and worldsize
            super().setup()

            # Initialize shard state manager with adjusted rank/worldsize
            self._shard_manager = ShardStateManager(
                n_logical_shards=self.n_logical_shards,
                rank=self.rank,
                worldsize=self.worldsize,
                field_enum=TitanMMShardField,
                seed=self.seed,
            )
            self._shard_manager.initialize()

            # Add packer tracker entries
            print(f".   Rank {self.rank} shards {self.shard_states[:,TitanMMShardField.SHARD_ID]}")
            self.packer_buffers = {i:[] for i in self.shard_states[:,TitanMMShardField.SHARD_ID]}
            self.packer_samples = {i:[] for i in self.shard_states[:,TitanMMShardField.SHARD_ID]}
    
    @property
    def shard_states(self) -> torch.Tensor:
        """
        Access the shard states tensor.

        This property provides backward compatibility for code that accesses
        shard_states directly, while delegating to the ShardStateManager.
        """
        if self._shard_manager is None:
            return None
        return self._shard_manager.state

    @shard_states.setter
    def shard_states(self, value: torch.Tensor) -> None:
        """
        Set the shard states tensor.

        This is called during checkpoint loading to restore the state.
        """
        if self._shard_manager is not None:
            self._shard_manager.state = value

    def construct_reader(self, rank, nshards):
        """
        TODO
        """
        # Map rank to underlying shuffled index
        datarank = self._shard_manager.get_shuffled_shard_id(rank)
        
        # Fetch relevant Titan data shard
        reader = self.data_constructor(dp_rank=datarank, dp_world_size=nshards)
        print(f".   Rank {self.rank} fetching shard {rank}")
        reader._sample_idx = self._shard_manager.get_titan_sample_idx(rank)
        if hasattr(reader, "packer"):
            reader.packer.sample_buffer.clear()
            reader.packer.packed_samples.clear()
            if rank not in self.packer_buffers:
                self.packer_buffers[rank] = []
                self.packer_samples[rank] = []
            reader.packer.sample_buffer.extend(self.packer_buffers[rank])
            reader.packer.packed_samples.extend(self.packer_samples[rank])
        self.current_stream = reader

    def __iter__(self):
        self.setup()
        reader = None
        epochs_without_yielding = 0
        assert len(self.shard_states) > 0 and self._shard_manager.has_valid_shards(), f"Worker {self.rank} of {self.worldsize} in {self.datapath} owns no logical shards!"
        while True:
            has_yielded = False
            # Isolate undervisited shards using epoch count field of shard_states
            epoch_count = self._shard_manager.get_min_epoch()
            shardset = self._shard_manager.get_shards_with_epoch(epoch_count)
            for j,k in enumerate(shardset):
                # Account for the relocation of each active shard_state
                # to the end of self.shard_states after it is exhausted
                i = k-j
                shardid = self._shard_manager.get_shard_id(i)
                self.construct_reader(i, self.n_logical_shards)
                reader = iter(self.current_stream)
                # For each shard, iterate through all the remaining docs
                self.current_shard = i
                while True:
                    try:
                        yield next(reader)
                        has_yielded = True

                        print(f".   Rank {self.rank} yielded an item!")
                    except StopIteration:
                        break
                # When shard is complete, reset state and clear position tracker
                self._shard_manager.set_titan_sample_idx(i, 0)
                # Update packer states to account for any overflow
                if hasattr(self.current_stream, "packer"):
                    self.packer_buffers[shardid] = list(self.current_stream.packer.sample_buffer)
                    self.packer_samples[shardid] = list(self.current_stream.packer.packed_samples)
                # Increase epoch count after finishing shard
                self._shard_manager.increment_epoch(i)
                # Prioritize unseen data after rescaling by shifting completed shard to end of shard_states
                # i.e. shards with (id, epoch_count) [(0,0),(1,1),(2,1),(3,2)] wll produce order:
                # 0,1,2,0,3,1,2,0,... instead of 0,0,1,2,0,1,2,3,...
                self._shard_manager.move_shard_to_end(i)
            if not has_yielded:
                epochs_without_yielding += 1

            print(f".   Rank {self.rank} finished an epoch!")
            
            # Begin new epoch, and verify that after visiting all shards, some data has been produced
            assert epochs_without_yielding < 3 or len(shardset)!=self._shard_manager.count_valid_shards(), f"Worker {self.rank} of {self.worldsize} in {self.datapath} owns no documents! {self.shard_states}"

    def state_dict(self):
        # Write current reader's state into shard state, and packer into packer trackers
        if self.current_shard != -1:
            self._shard_manager.set_titan_sample_idx(self.current_shard, self.current_stream._sample_idx)
            if hasattr(self.current_stream, "packer"):
                self.packer_buffers[self.current_shard] = list(self.current_stream.packer.sample_buffer)
                self.packer_samples[self.current_shard] = list(self.current_stream.packer.packed_samples)
        return super().state_dict()
        

class ScalableHFReader(_StatefulDataset):
    """
    TODO
    """

    def __init__(
        self,
        datapath: str,
        rank: int,
        worldsize: int,
        tokenizer: Tokenizer,
        delimiter_token: Any,
        bos_token: Optional[Any] = None,
        strip_tokens: Optional[Set[Any]] = set(),
        min_length: int = 1,
        col_names: List[str] = ["text", "contents", "tokens"],
        n_logical_shards: int = 30720,
        seed: int = 42,
    ):
        super().__init__(datapath, rank, worldsize)
        self.datapath = datapath
        self.tokenizer = tokenizer
        self.min_length = min_length  # Ignore any docs shorter than this
        self.eos = delimiter_token  # Inserted between each doc
        self.bos = bos_token  # Inserted before each doc (optional)
        self.drop = strip_tokens  # Tokens to drop from begin/end of doc (replaced by above delimiter/bos)
        self.col_names = col_names  # For each data point, grab first field that matches an entry in this list
        self.n_logical_shards = n_logical_shards
        self.seed = seed

        # Setup flags
        self.stream = None

        # Position
        self.current_shard = -1
        self.current_stream = None

        # Shard state manager (initialized in setup)
        self._shard_manager: Optional[ShardStateManager] = None

        self.custom_vars = ["shard_states"]
        self.custom_fns = [lambda shard_states: shard_rescale(shard_states, self.rank, self.worldsize)]

    @property
    def shard_states(self) -> torch.Tensor:
        """
        Access the shard states tensor.

        This property provides backward compatibility for code that accesses
        shard_states directly, while delegating to the ShardStateManager.
        """
        if self._shard_manager is None:
            return None
        return self._shard_manager.state

    @shard_states.setter
    def shard_states(self, value: torch.Tensor) -> None:
        """
        Set the shard states tensor.

        This is called during checkpoint loading to restore the state.
        """
        if self._shard_manager is not None:
            self._shard_manager.state = value

    def setup(self):
        """
        Perform any rank- and path-dependent setup. This operation is deferred from __init__
        to support multiple workers in the dataloader.
        """
        if not self.is_setup:
            # Get your adjusted rank and worldsize
            super().setup()

            # Initialize shard state manager with adjusted rank/worldsize
            self._shard_manager = ShardStateManager(
                n_logical_shards=self.n_logical_shards,
                rank=self.rank,
                worldsize=self.worldsize,
                field_enum=HFShardField,
                seed=self.seed,
            )
            self._shard_manager.initialize()

            # Open HF stream
            path, name = os.path.split(self.datapath)
            self.stream = load_dataset(path, name=name, split="train", streaming=True)

    def construct_reader(self, rank, nshards, shard_state):
        """
        TODO
        """
        # Map rank to underlying shuffled index
        rank = self._shard_manager.get_shuffled_shard_id(rank)
        # Fetch relevant HF data shard
        reader = split_dataset_by_node(self.stream, rank, nshards)
        d = reader.state_dict()
        d['examples_iterable']['examples_iterable']['shard_idx'] = shard_state[HFShardField.SHARD_IDX].item()
        d['examples_iterable']['examples_iterable']['shard_example_idx'] = shard_state[HFShardField.SHARD_EXAMPLE_IDX].item()
        reader.load_state_dict(d)
        self.current_stream = reader

    def _process_doc(self, data):
        """
        Tokenize doc and handle bos/eos
        """
        # Pull out relevant text field
        doc = None
        for name in self.col_names:
            if name in data.keys():
                doc = data[name]
                break
        assert (
            doc is not None
        ), f"None of column names {self.col_names} found in file headers {data.keys()}"
        # Tokenize
        doc = self.tokenizer.encode(doc)
        # Truncate first token if needed
        if len(doc) > 0 and doc[0] in self.drop:
            doc = doc[1:]
        # Recheck len for edge case where doc=[eos]
        if len(doc) > 0 and doc[-1] in self.drop:
            doc = doc[:-1]
        # Add bos/eos tokens
        if self.bos is not None:
            doc = [self.bos] + doc
        doc = doc + [self.eos]
        return doc

    def __iter__(self):
        self.setup()
        reader = None
        has_yielded = False
        assert len(self.shard_states) > 0 and self._shard_manager.has_valid_shards(), f"Worker {self.rank} of {self.worldsize} in {self.datapath} owns no logical shards!"
        while True:
            # Isolate undervisited shards using epoch count field of shard_states
            epoch_count = self._shard_manager.get_min_epoch()
            shardset = self._shard_manager.get_shards_with_epoch(epoch_count)
            for j,k in enumerate(shardset):
                # Account for the relocation of each active shard_state
                # to the end of self.shard_states after it is exhausted
                i = k-j
                shardid = self._shard_manager.get_shard_id(i)
                self.construct_reader(shardid, self.n_logical_shards, self.shard_states[i])
                reader = iter(self.current_stream)
                # For each shard, iterate through all the remaining docs
                self.current_shard = i
                while True:
                    try:
                        doc = next(reader)
                        seq = self._process_doc(doc)
                        yield seq
                        has_yielded = True
                    except StopIteration:
                        break
                # When shard is complete, reset state and clear position tracker
                self._shard_manager.set_hf_shard_idx(i, 0)
                self._shard_manager.set_hf_shard_example_idx(i, 0)
                # Increase epoch count after finishing shard
                self._shard_manager.increment_epoch(i)
                # Prioritize unseen data after rescaling by shifting completed shard to end of shard_states
                # i.e. shards with (id, epoch_count) [(0,0),(1,1),(2,1),(3,2)] wll produce order:
                # 0,1,2,0,3,1,2,0,... instead of 0,0,1,2,0,1,2,3,...
                self._shard_manager.move_shard_to_end(i)
            # Begin new epoch, and verify that after visiting all shards, some data has been produced
            assert has_yielded or len(shardset)!=self._shard_manager.count_valid_shards(), f"Worker {self.rank} of {self.worldsize} in {self.datapath} owns no documents! {self.shard_states}"

    def state_dict(self):
        # Write current reader's state into shard state
        if self.current_shard != -1:
            d = self.current_stream.state_dict()
            self._shard_manager.set_hf_shard_idx(
                self.current_shard,
                d['examples_iterable']['examples_iterable']['shard_idx']
            )
            self._shard_manager.set_hf_shard_example_idx(
                self.current_shard,
                d['examples_iterable']['examples_iterable']['shard_example_idx']
            )
        return super().state_dict()

class ScalableReader(_StatefulDataset):
    """
    Iterates through a shard of all data in the specified datapath, as determined by rank and worldsize.
    Implements rescalability by dividing data into a large number of logical shards, and allocating
    logical shards over physical dataloader workers. During iteration, shards with the lowest visit
    count are exhausted before continuing into new a epoch. When rescaling, re-allocates logical shards
    to achieve as even data coverage as possible, so that each new worker finishes its epoch at around
    the same time.

    Local state is an n x 5 matrix where n is the number of logical shards owned by this worker, and 5
    is the number of relevant data fields per-shard. The 5 fields are: shard index, file index,
    document index, document chunk index, and visitation/epoch count. This information, aggregated
    across workers, is sufficient to track the entirety of seen and unseen data in the dataset.

    Currently does not shuffle docs within shards/files, but this can be added later.
    ...
    Args
    ----
    datapath : str
        Absolute path to a directory containing data files. Directory need not be flat: all files under
        the current path will be detected so long as they are determined valid by the filehandler.
    rank : int
        Rank of the current device w.r.t. data parallelism.
    worldsize : int
        Total number of devices w.r.t. data parallelism.
    filehandler : file_handlers.ShardFileHandler
        A FileHandler used to detect and interface with the data files in the datapath.
    delimiter_token : Any
        A token inserted at the end of each retrieved sequence/document. Indicates end of document
        for subsequent wrappers / loader stages (i.e. packing/slicing, shuffling, subdataset sampling).
        If not needed, can be removed in subsequent stages instead. Data type should match the
        underlying data sequences being loaded.
    bos_token : Any
        An optional token inserted at the beginning of each retrieved sequence/document. Data type
        should match the underlying data sequences being loaded. Note that specifying this and
        delimiter_token will result in paired delimiters when documents are packed together,
        i.e. <doc> <delimiter> <bos> <doc>
    strip_tokens : Set[Any]
        A set of values to be removed from loaded sequences/documents, if they occur in the first or
        last position. Used to remove any existing bos/eos/delimiter tokens before inserting the
        specified bos/delimiter above.
    min_length : int
        Any loaded sequences/documents shorter than this value will be skipped.
    max_chunksize : int
        If a loaded sequence/document is longer than this value, it will instead be partitioned into
        chunks of this size or smaller (not counting added bos/delimiter tokens), which are emitted
        in order. For pyarrow data file formats, this prevents extra overhead when loading an extremely
        long sequence/document.
    n_logical_shards : int
        The number of logical data partitions. This value should be much larger than the number of
        dataloader workers, and also much smaller than the number of sequences/documents in the dataset.
        This ensures that workers exhaust their data and finish their epochs at roughly the same time.
    seed : int
        Random seed used to shuffle logical shard assignments
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
        min_length: int = 1,
        max_chunksize: int = 1024,
        n_logical_shards: int = 30720,
        seed: int = 42,
    ):
        super().__init__(datapath, rank, worldsize)
        assert datapath is None or (
            os.path.isdir(datapath) and len(os.listdir(datapath)) > 0
        ), f"Data path {datapath} must be a non-empty folder or None"
        self.datapath = datapath
        self.filehandler = filehandler
        self.min_length = min_length  # Ignore any docs shorter than this
        assert max_chunksize > 0, f"Max chunksize must be a nonzero positive integer"
        self.chunksize = max_chunksize  # Yield chunks at a time if doc is longer than this
        self.eos = delimiter_token  # Inserted between each doc
        self.bos = bos_token  # Inserted before each doc (optional)
        self.drop = strip_tokens  # Tokens to drop from begin/end of doc (replaced by above delimiter/bos)
        self.n_logical_shards = n_logical_shards
        self.seed = seed

        # Position
        self.reader = None
        self.cur_file = None

        # Setup flags
        self.filesizes = None  # [[filenames], [filesizes]]  (constructed pre-iter if not loaded from ckp)

        # Shard state manager (initialized in setup)
        self._shard_manager: Optional[ShardStateManager] = None

        self.broadcast_vars = ["filesizes"]
        self.custom_vars = ["shard_states"]
        self.custom_fns = [lambda shard_states: shard_rescale(shard_states, self.rank, self.worldsize)]

    @property
    def shard_states(self) -> torch.Tensor:
        """
        Access the shard states tensor.

        This property provides backward compatibility for code that accesses
        shard_states directly, while delegating to the ShardStateManager.
        """
        if self._shard_manager is None:
            return None
        return self._shard_manager.state

    @shard_states.setter
    def shard_states(self, value: torch.Tensor) -> None:
        """
        Set the shard states tensor.

        This is called during checkpoint loading to restore the state.
        """
        if self._shard_manager is not None:
            self._shard_manager.state = value

    def _get_shard_breakdown(self, shard_id, nshards):
        """
        Retrieve the set of (fractional) files assigned to a given logical shard. Returns a list of
        data files, indicating for each file: the file index, and the start and end points, expressed
        as percentage points of the entire file.
        """
        # Map shard_id to underlying shuffled index
        # TODO1: why do we need this ? does this enable shuffling ?
        # TODO2: we return file_ids here, this assumes the system can pull up random docs based on int file_id
        # TODO3: since this probably enables shuffle, we should see how we are breaking the user's expectation of shuffle,
        #    for eg if user has a specific shuffling logic, we are breaking it here,
        #    these are iterable datasets but probably worth thinking about it.
        # shuffle in the algorithm - not default in the system unless asked for.
        shuffled_shard_id = self._shard_manager.get_shuffled_shard_id(shard_id) # can make this optional! also, on file sort in 546
        # Find first doc included in the current shard
        sizelist = torch.tensor(self.filesizes[1])
        sizelist = sizelist/sizelist.float().sum()
        cum_sizelist = sizelist.cumsum(0)
        start_frac = shuffled_shard_id/nshards
        file_start_id = len(sizelist) - cum_sizelist.gt(start_frac).sum().item()
        # For each doc, assign relevant fractional ownership
        start = start_frac
        end = (shuffled_shard_id+1)/nshards
        my_files = []  # fileid, start%, end%
        for i, (size, cumsize_incl) in enumerate(
            zip(sizelist[file_start_id:].tolist(), cum_sizelist[file_start_id:].tolist())
        ):
            file_id = file_start_id + i
            cumsize = cumsize_incl - size
            if cumsize > end:
                # No more files to include, stop early
                break
            elif cumsize <= end and cumsize_incl >= start:
                my_files.append([
                    file_id,
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

            # Check that datapath, post setup, is still legal
            assert os.path.isdir(self.datapath) and len(os.listdir(self.datapath)) > 0, f"Invalid dataset {self.datapath}"

            # Initialize shard state manager with adjusted rank/worldsize
            self._shard_manager = ShardStateManager(
                n_logical_shards=self.n_logical_shards,
                rank=self.rank,
                worldsize=self.worldsize,
                field_enum=ShardField,
                seed=self.seed,
            )
            self._shard_manager.initialize()

    def _pre_iter(self):
        """
        Construct index of data files and their filesizes. This is saved/loaded in subsequent
        checkpoints to avoid re-indexing the entire dataset repeatedly (and so deferred from
        self.setup to ensure that this only runs AFTER loading a given checkpoint).
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
        self.setup()
        self._pre_iter()
        reader = None
        ndocs = -1
        has_yielded = False
        assert len(self.shard_states) > 0 and self._shard_manager.has_valid_shards(), f"Worker {self.rank} of {self.worldsize} in {self.datapath} owns no logical shards!"
        while True:
            # Isolate undervisited shards using epoch count field of shard_states
            epoch_count = self._shard_manager.get_min_epoch()
            shardset = self._shard_manager.get_shards_with_epoch(epoch_count)
            for j,k in enumerate(shardset):
                # Account for the relocation of each active shard_state
                # to the end of self.shard_states after it is exhausted
                i = k-j
                shardid = self._shard_manager.get_shard_id(i)
                files = self._get_shard_breakdown(shardid, self.n_logical_shards)  # list([docid, start%, end%])
                file_offset = self._shard_manager.get_file_pos(i)
                # For each shard, iterate over the contained data files, starting from any specified offset
                for file_pos in range(file_offset, len(files)):
                    # Update position
                    self._shard_manager.set_file_pos(i, file_pos)
                    # Calculate doc range
                    file = files[file_pos]
                    fileid = file[0]
                    reader, ndocs = self._get_reader(fileid, reader, ndocs)
                    doc_start = round(ndocs * file[1])
                    doc_end = round(ndocs * file[2])
                    doc_offset = self._shard_manager.get_doc_pos(i)
                    # For each file, iterate over the specified range of contained sequences/documents
                    for doc_pos in range(doc_offset, doc_end - doc_start):
                        # Update position
                        self._shard_manager.set_doc_pos(i, doc_pos)
                        # Fetch doc
                        doc = self.filehandler.get(reader, doc_start + doc_pos, self.drop)
                        doclen = len(doc)
                        nchunks = math.ceil(doclen/self.chunksize)
                        chunk_offset = self._shard_manager.get_chunk_pos(i)
                        # For each sequence/document, iterate over the chunks to emit
                        for chunk_pos in range(chunk_offset, nchunks):
                            # Update position
                            self._shard_manager.set_chunk_pos(i, chunk_pos+1)
                            # Yield chunk
                            yield self._construct_chunk(chunk_pos, doc, nchunks)
                            has_yielded = True
                        # Reset chunk_pos after finishing doc
                        self._shard_manager.set_chunk_pos(i, 0)
                    # Reset doc_pos after finishing file
                    self._shard_manager.set_doc_pos(i, 0)
                # Reset file_pos after finishing shard
                self._shard_manager.set_file_pos(i, 0)
                # Increase epoch count after finishing shard
                self._shard_manager.increment_epoch(i)
                # Prioritize unseen data after rescaling by shifting completed shard to end of shard_states
                # i.e. shards with (id, epoch_count) [(0,0),(1,1),(2,1),(3,2)] wll produce order:
                # 0,1,2,0,3,1,2,0,... instead of 0,0,1,2,0,1,2,3,...
                self._shard_manager.move_shard_to_end(i)
            # Begin new epoch, and verify that after visiting all shards, some data has been produced
            assert has_yielded or len(shardset)!=self._shard_manager.count_valid_shards(), f"Worker {self.rank} of {self.worldsize} in {self.datapath} owns no documents! {self.shard_states}"


