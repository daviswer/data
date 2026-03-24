"""
DCP checkpoint saving utilities.

Handles distributed saving of dataloader state dicts using PyTorch DCP.
State dict values are processed according to their tag type (state, broadcast,
reshard, custom) before saving.
"""

import os
from copy import deepcopy
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.distributed import checkpoint

from .dtensor_utils import wrap_dtensor, wrap_shardtensor


def save_ckpt_dcp(
    loader: Any,
    path: str,
    device_mesh: dist.DeviceMesh,
) -> None:
    """
    Retrieves dataloader state dict, and separates worker states from loader state.
    Aggregates worker states, and wraps/processes state/broadcast/reshard/custom variables.
    Saves states concurrently through DCP APIs.

    Args:
        loader: StatefulDataLoader instance to save
        path: Directory path to save checkpoint
        device_mesh: Device mesh for distributed tensors
    """
    os.makedirs(path, exist_ok=True)
    # TODO: do we have to get rank and worldsize from loader.dataset? make it get from loader instead?
    rank = loader.dataset.rank
    worldsize = loader.dataset.worldsize
    # TODO: this deepcopy can be expensive if the dataset state is large, consider an optimization if needed
    state = deepcopy(loader.state_dict())
    # TODO: Make it flexible to support num_workers = 0
    nworkers = state["_snapshot"]["_main_snapshot"]["_num_workers"]

    # Extract and restructure worker dataset states
    dstate = _extract_worker_states(state, nworkers)

    # Process each state category
    state_vars = _prepare_state_vars(dstate, state, device_mesh, rank, worldsize, nworkers)
    dstate["state"] = state_vars

    _prepare_broadcast_vars(dstate, rank, worldsize, nworkers)
    _prepare_reshard_vars(dstate, state_vars, device_mesh, rank, worldsize)
    _prepare_custom_vars(dstate, rank, nworkers)

    checkpoint.save(
        dstate,
        storage_writer=checkpoint.FileSystemWriter(path=path),
        planner=checkpoint.DefaultSavePlanner(),
    )


def _extract_worker_states(state: Dict, nworkers: int) -> Dict[str, List[Dict]]:
    """
    Extract and restructure worker dataset states from loader state dict.

    Flips List[dict[dict]] to dict[List[dict]] for easier processing.

    Args:
        state: Full loader state dict
        nworkers: Number of workers

    Returns:
        Dictionary with keys {state, broadcast, reshard, custom}, each
        containing a list of per-worker dictionaries
    """
    dstate = state["_snapshot"]["_worker_snapshots"]
    dstate = [
        dstate[f"worker_{i}"].pop("dataset_state") for i in range(len(dstate))
    ]
    # Flip List[dict[dict]] to dict[List[dict]]
    dstate = {k: [d[k] for d in dstate] for k in dstate[0].keys()}
    return dstate


def _prepare_state_vars(
    dstate: Dict,
    loader_state: Dict,
    device_mesh: dist.DeviceMesh,
    rank: int,
    worldsize: int,
    nworkers: int,
) -> Dict:
    """
    Prepare state variables for saving.

    State variables are wrapped in DTensors. The loader state and reshard sizes
    are added as additional metadata.

    Args:
        dstate: Worker states dictionary
        loader_state: Full loader state dict
        device_mesh: Device mesh for distributed tensors
        rank: Current rank
        worldsize: Total ranks
        nworkers: Number of workers

    Returns:
        Processed state variables wrapped in DTensors
    """
    state_vars = dstate["state"]
    # Flip List[dict] to dict[List]
    state_vars = {k: [d[k] for d in state_vars] for k in state_vars[0].keys()}
    state_vars["loader_state"] = loader_state
    # reshard_sizes will be added by _prepare_reshard_vars before wrapping
    return state_vars


def _prepare_broadcast_vars(
    dstate: Dict,
    rank: int,
    worldsize: int,
    nworkers: int,
) -> None:
    """
    Prepare broadcast variables for saving.

    Only saves from rank 0. Adds checkpoint worldsize metadata.

    Args:
        dstate: Worker states dictionary (modified in place)
        rank: Current rank
        worldsize: Total ranks
        nworkers: Number of workers
    """
    broadcast_vars = dstate.pop("broadcast")[0]
    if rank == 0:
        dstate["broadcast"] = broadcast_vars
        dstate["broadcast"]["global_worldsize"] = worldsize * nworkers


def _prepare_reshard_vars(
    dstate: Dict,
    state_vars: Dict,
    device_mesh: dist.DeviceMesh,
    rank: int,
    worldsize: int,
) -> None:
    """
    Prepare reshard variables for saving.

    Concatenates entries, fetches global size, wraps in sharding DTensor,
    and stores partial sizes in state_vars for use when not rescaling.

    Args:
        dstate: Worker states dictionary (modified in place)
        state_vars: State variables dict (modified to add reshard_sizes)
        device_mesh: Device mesh for distributed tensors
        rank: Current rank
        worldsize: Total ranks
    """
    reshard_vars = dstate["reshard"]

    # Assert all reshard vals are tensors
    for k, v in reshard_vars[0].items():
        assert isinstance(v, torch.Tensor), f"Reshard var {k} is not a torch tensor!"

    # Flip list[dict] to dict[list]
    reshard_vars = {k: [d[k] for d in reshard_vars] for k in reshard_vars[0].keys()}

    # Inject per-worker shard sizes into state_vars for use when not rescaling
    state_vars["reshard_sizes"] = {
        k: [x.size(0) for x in v] for k, v in reshard_vars.items()
    }

    # Now wrap state_vars in DTensors (after reshard_sizes is added)
    wrapped_state = wrap_dtensor(state_vars, device_mesh)
    dstate["state"] = wrapped_state

    # Concat and wrap in DTensor
    reshard_vars = {
        k: wrap_shardtensor(torch.cat(v, dim=0), device_mesh, rank, worldsize)
        for k, v in reshard_vars.items()
    }
    dstate["reshard"] = reshard_vars


def _prepare_custom_vars(dstate: Dict, rank: int, nworkers: int) -> None:
    """
    Prepare custom variables for saving.

    Prepends rank to every key to prevent collisions across workers.

    Args:
        dstate: Worker states dictionary (modified in place)
        rank: Current rank
        nworkers: Number of workers
    """
    custom_vars = dstate["custom"]
    # Convert list[dict] to dict with prepended rank in keys
    custom_vars = {
        f"rank{rank * nworkers + i}.{k}": custom_vars[i][k]
        for i in range(len(custom_vars))
        for k in custom_vars[0].keys()
    }
    dstate["custom"] = custom_vars
