"""
Shard rescaling utilities for ScalableReader and ScalableHFReader.
The states for these classes require more sophisticated handling than DCP's default DTensor sharding,
so we handle resharding of custom state variables here.

When rescaling to a different number of workers, custom state variables are aggregated globally onto
each rank, and rely on these methods to reshard/extract the appropriate state for a given rank. 

As such, these functions are called ONLY when rescaling.
"""

import pickle
from typing import Any,Callable,List

import torch

from .shard_state import DUMMY_EPOCH, DUMMY_SHARD_ID


def atomic_rescale(
        shard_states: List[Any], 
        rank: int, 
        worldsize: int, 
    ) -> List[Any]:
    """
    Performs resharding of non-splittable, atomic states, redistributing over available ranks. 
    Note that ranks may recieve no, or multiple, states after calling this function.
    It is up to the caller to provide default values for empty lists, extract the state from
    singleton lists, or merge entries in multi-item lists.
    """
    if len(shard_states) == worldsize:
        # This covers the case where the number of gpus changes, but the number of dataloader
        # workers does not. In this case, simply pull out the corresponding rank.
        return [shard_states[rank]]
    else:
        n_items = len(shard_states)
        start = (rank*n_items)//worldsize
        end = (rank*n_items+n_items)//worldsize
        state = shard_states[start:end]
        print(f"Rank {rank}: {end-start, n_items}")
        return state


def epoch_balanced_rescale(shard_states: List[torch.Tensor], rank: int, worldsize: int) -> torch.Tensor:
    """
    Custom function for rescaling of ScalableReader / HFReader shard_states. Logical shards,
    aggregated across workers, are split based on whether they have been visited in the current
    epoch, and each partition is re-allocated across the new worker set such that each new worker
    receives the same number of visited, unvisited, and total shards (at most off by one). This
    ensures that each worker receives roughly the same ratio of seen/unseen data in the current epoch.
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
