"""
Named constants and enums for shard state management.

The shard_states tensor is an n x k matrix where:
- n = number of logical shards owned by this worker
- k = number of state fields per shard

This module provides named constants to replace magic indices when accessing
shard state fields, improving code readability.
"""

import math
from enum import IntEnum
from typing import List, Optional

import torch


# Sentinel values for dummy/padding shards
DUMMY_SHARD_ID = -1
DUMMY_EPOCH = torch.iinfo(torch.int).max


class ShardField(IntEnum):
    """
    Column indices for ScalableReader shard_states tensor.

    The 5 fields are: shard index, file index, document index,
    document chunk index, and visitation/epoch count.

    This information, aggregated across workers, is sufficient to track
    the entirety of seen and unseen data in the dataset.
    """

    SHARD_ID = 0
    FILE_POS = 1
    DOC_POS = 2
    CHUNK_POS = 3
    EPOCH = 4


class HFShardField(IntEnum):
    """
    Column indices for ScalableHFReader shard_states tensor.

    Uses a slightly different schema than ScalableReader since HuggingFace
    datasets handle file/doc position internally via their state_dict.
    """

    SHARD_ID = 0
    SHARD_IDX = 1
    SHARD_EXAMPLE_IDX = 2
    EPOCH = 3


class ShardStateManager:
    """
    Manages the state matrix for logical shards with readable accessors.

    The state matrix is an n x k tensor where:
    - n = number of logical shards owned by this worker (ceil(n_logical_shards / worldsize))
    - k = number of state fields per shard (determined by field_enum)

    This class provides:
    - Initialization with round-robin shard allocation
    - Readable accessors for state fields (replacing cryptic tensor operations)
    - Methods for common operations like counting valid shards, getting min epoch, etc.

    The underlying state tensor is accessible via the `state` property for backward
    compatibility with existing code that directly manipulates `shard_states`.
    """

    def __init__(
        self,
        n_logical_shards: int,
        rank: int,
        worldsize: int,
        field_enum: type,
        seed: int = 42,
    ):
        """
        Initialize the shard state manager.

        Args:
            n_logical_shards: Total number of logical shards across all workers
            rank: This worker's rank
            worldsize: Total number of workers
            field_enum: The enum class defining state fields (ShardField or HFShardField)
            seed: Random seed for shuffling shard assignments
        """
        self.n_logical_shards = n_logical_shards
        self.rank = rank
        self.worldsize = worldsize
        self.field_enum = field_enum
        self.n_fields = len(field_enum)
        self.seed = seed

        # Shuffle for randomizing shard assignments
        self.shuffle = torch.randperm(
            n_logical_shards, generator=torch.Generator().manual_seed(seed)
        )

        # State tensor (initialized in initialize())
        self._state: Optional[torch.Tensor] = None

    @property
    def state(self) -> torch.Tensor:
        """Get the underlying state tensor."""
        return self._state

    @state.setter
    def state(self, value: torch.Tensor) -> None:
        """Set the underlying state tensor (for checkpoint loading)."""
        self._state = value

    def initialize(self) -> None:
        """
        Allocate shards to this worker using round-robin and create state matrix.

        Uses round-robin allocation to facilitate order preservation during rescaling.
        Pads with dummy shards if this worker has fewer shards than others.
        """
        # Get logical shard partitions. Use round-robin allocation to facilitate
        # order preservation during rescaling
        my_shards = self._compute_shard_allocation()

        # Set up logical shard states (may be overwritten later by ckp load)
        n_rows = math.ceil(self.n_logical_shards / self.worldsize)
        self._state = torch.zeros(n_rows, self.n_fields, dtype=torch.int)

        # Set shard ids
        self._state[: len(my_shards), self.field_enum.SHARD_ID] = torch.tensor(my_shards)

        # Pad shard state if this worker is off by one. Id is -1 and visit count is inf.
        self._state[len(my_shards) :, self.field_enum.SHARD_ID] = DUMMY_SHARD_ID
        self._state[len(my_shards) :, -1] = DUMMY_EPOCH  # epoch field is always last

    def _compute_shard_allocation(self) -> List[int]:
        """
        Compute round-robin allocation of logical shards to this worker.

        Returns:
            List of logical shard IDs assigned to this worker
        """
        n_shards = self.n_logical_shards // self.worldsize + int(
            self.rank < self.n_logical_shards % self.worldsize
        )
        return [
            (x * self.worldsize + self.rank) % self.n_logical_shards
            for x in range(n_shards)
        ]

    def has_valid_shards(self) -> bool:
        """
        Check if this worker owns at least one valid (non-dummy) logical shard.

        A shard is valid if its ID is not -1 (DUMMY_SHARD_ID).
        """
        return (self._state[:, self.field_enum.SHARD_ID] != DUMMY_SHARD_ID).sum() > 0

    def count_valid_shards(self) -> int:
        """
        Count shards with ID > 0 (excludes shard 0 and dummy shards).

        This is used in assertions to verify data was produced.
        """
        return (self._state[:, self.field_enum.SHARD_ID] > 0).sum().item()

    def count_non_dummy_shards(self) -> int:
        """
        Count all non-dummy shards (includes shard 0).
        """
        return (self._state[:, self.field_enum.SHARD_ID] != DUMMY_SHARD_ID).sum().item()

    def get_min_epoch(self) -> int:
        """
        Get the minimum epoch count across all shards.

        Used to find undervisited shards that should be processed first.
        """
        return self._state[:, -1].min().item()

    def get_shards_with_epoch(self, epoch: int) -> torch.Tensor:
        """
        Get indices of shards that have the specified epoch count.

        Used to isolate undervisited shards for processing.

        Args:
            epoch: The epoch count to filter by

        Returns:
            Tensor of indices into state matrix
        """
        return self._state[:, -1].eq(epoch).nonzero().squeeze(-1)

    def get_shard_id(self, idx: int) -> int:
        """Get the logical shard ID at the given index."""
        return self._state[idx, self.field_enum.SHARD_ID].item()

    def get_shuffled_shard_id(self, shard_id: int) -> int:
        """
        Map a logical shard ID to its shuffled index.

        Used to randomize shard-to-file mapping while maintaining determinism.
        """
        return self.shuffle[shard_id].item()

    # ─────────────────────────────────────────────────────────────
    # Position tracking (for ScalableReader)
    # ─────────────────────────────────────────────────────────────

    def get_file_pos(self, idx: int) -> int:
        """Get the current file position for shard at given index."""
        return self._state[idx, ShardField.FILE_POS].item()

    def set_file_pos(self, idx: int, pos: int) -> None:
        """Set the current file position for shard at given index."""
        self._state[idx, ShardField.FILE_POS] = pos

    def get_doc_pos(self, idx: int) -> int:
        """Get the current document position for shard at given index."""
        return self._state[idx, ShardField.DOC_POS].item()

    def set_doc_pos(self, idx: int, pos: int) -> None:
        """Set the current document position for shard at given index."""
        self._state[idx, ShardField.DOC_POS] = pos

    def get_chunk_pos(self, idx: int) -> int:
        """Get the current chunk position for shard at given index."""
        return self._state[idx, ShardField.CHUNK_POS].item()

    def set_chunk_pos(self, idx: int, pos: int) -> None:
        """Set the current chunk position for shard at given index."""
        self._state[idx, ShardField.CHUNK_POS] = pos

    def get_epoch(self, idx: int) -> int:
        """Get the epoch count for shard at given index."""
        return self._state[idx, -1].item()

    def increment_epoch(self, idx: int) -> None:
        """Increment the epoch count for shard at given index."""
        self._state[idx, -1] += 1

    def reset_position(self, idx: int) -> None:
        """
        Reset file/doc/chunk positions to 0 after completing a shard.

        Only resets position fields, not shard_id or epoch.
        """
        # Reset all position fields (all fields between shard_id and epoch)
        for field_idx in range(1, self.n_fields - 1):
            self._state[idx, field_idx] = 0

    # ─────────────────────────────────────────────────────────────
    # HuggingFace-specific position tracking (for ScalableHFReader)
    # ─────────────────────────────────────────────────────────────

    def get_hf_shard_idx(self, idx: int) -> int:
        """Get the HF shard_idx for shard at given index."""
        return self._state[idx, HFShardField.SHARD_IDX].item()

    def set_hf_shard_idx(self, idx: int, value: int) -> None:
        """Set the HF shard_idx for shard at given index."""
        self._state[idx, HFShardField.SHARD_IDX] = value

    def get_hf_shard_example_idx(self, idx: int) -> int:
        """Get the HF shard_example_idx for shard at given index."""
        return self._state[idx, HFShardField.SHARD_EXAMPLE_IDX].item()

    def set_hf_shard_example_idx(self, idx: int, value: int) -> None:
        """Set the HF shard_example_idx for shard at given index."""
        self._state[idx, HFShardField.SHARD_EXAMPLE_IDX] = value

    # ─────────────────────────────────────────────────────────────
    # Shard reordering (for prioritizing unseen data after rescaling)
    # ─────────────────────────────────────────────────────────────

    def move_shard_to_end(self, idx: int) -> None:
        """
        Move completed shard to end of state matrix.

        This prioritizes unseen data after rescaling by shifting completed shards
        to the end of shard_states.

        Example: shards with (id, epoch_count) [(0,0),(1,1),(2,1),(3,2)] will produce order:
        0,1,2,0,3,1,2,0,... instead of 0,0,1,2,0,1,2,3,...
        """
        self._state = torch.cat(
            [
                self._state[:idx],
                self._state[idx + 1 :],
                self._state[idx : idx + 1],
            ],
            dim=0,
        )
