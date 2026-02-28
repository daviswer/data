"""
Shard rescaling utilities for ScalableReader and ScalableHFReader.

When rescaling to a different number of workers, the logical shard progress counters are aggregated
globally onto each ScalableReader. Then, completed and incomplete logical shards are re-allocated
separately, to ensure that each worker receives roughly the same ratio of seen to unseen data in the
current epoch. This allows us to scale from any number of workers to any other number.
"""

from typing import List

import torch

from .shard_state import DUMMY_EPOCH, DUMMY_SHARD_ID


def shard_rescale(shard_states: List[torch.Tensor], rank: int, worldsize: int) -> torch.Tensor:
    """
    Custom function for rescaling of ScalableReader / HFReader shard_states. Logical shards,
    aggregated across workers, are split based on whether they have been visited in the current
    epoch, and each partition is re-allocated across the new worker set such that each new worker
    receives the same number of visited, unvisited, and total shards (at most off by one).
    """
    if len(shard_states) == worldsize:
        # This covers the case where the number of gpus changes, but the number of dataloader
        # workers does not. In this case, simply pull out the corresponding rank.
        return shard_states[rank]
    else:
        # Sort shards by epoch count, then id
        shard_states = torch.cat(shard_states, dim=0)
        _, indices = torch.sort(shard_states[:, 0])
        shard_states = shard_states[indices]
        sorted_epochs, indices = torch.sort(shard_states[:, -1], descending=True, stable=True)
        shard_states = shard_states[indices]

        # Strip out dummy padding shards
        n_dummies = sorted_epochs.eq(DUMMY_EPOCH).sum()
        shard_states = shard_states[n_dummies:]  # n_logical x state_fields
        sorted_epochs = sorted_epochs[n_dummies:]

        # Split into max and non-max epochs
        n_complete = sorted_epochs.eq(sorted_epochs[0]).sum()
        completed_shards = shard_states[:n_complete]
        incomplete_shards = shard_states[n_complete:]

        # Re-allocate completed and incomplete shards round-robin, to loosely preserve ordering
        completed_shards = _reallocate_round_robin(completed_shards, worldsize)
        incomplete_shards = _reallocate_round_robin(incomplete_shards, worldsize)

        # Sort completed shards by length
        completed_shards.sort(key=len)

        # Reverse sort incomplete shards by length
        # Minimizes padding by overallocating incomplete shards to underallocated complete shards
        incomplete_shards.sort(key=len, reverse=True)

        # Pull out shard allocation for this worker
        # (sort/reverse-sort ensures allocations are off by no more than 1)
        shards = [completed_shards[rank], incomplete_shards[rank]]
        shard_states = torch.cat(shards)

        # Pad out with dummy shards if needed
        shard_states[len(shard_states) :, 0] = DUMMY_SHARD_ID
        shard_states[len(shard_states) :, -1] = DUMMY_EPOCH

        return shard_states


def _reallocate_round_robin(
    shard_states: torch.Tensor, worldsize: int
) -> List[torch.Tensor]:
    """
    Re-allocate shards round-robin across workers to loosely preserve ordering.

    Args:
        shard_states: Tensor of shard states to reallocate
        worldsize: Number of workers to distribute across

    Returns:
        List of tensors, one per worker, containing their allocated shards
    """
    return [
        shard_states[
            [
                (x * worldsize + r) % len(shard_states)
                for x in range(
                    len(shard_states) // worldsize
                    + int(r < len(shard_states) % worldsize)
                )
            ]
        ]
        for r in range(worldsize)
    ]
