"""
DCP checkpoint loading utilities.

Handles distributed loading of dataloader state dicts using PyTorch DCP.
Supports both same-scale loading and rescaling to different worker counts.
"""

import functools
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.distributed.tensor as dtensor
from torch.distributed import checkpoint
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor._shards_wrapper import LocalShardsWrapper

from .dtensor_utils import build_dtensor, crawl, unwrap_dtensor
from .metadata import get_checkpoint_metadata

# TODO: support for storing checkpoints in object storage
    # Titan should decide if this is needed

def load_ckpt_dcp(
    loader: Any,
    path: str,
    device_mesh: dist.DeviceMesh,
) -> None:
    """
    Retrieves dataloader state dict, and separates worker states from loader state.
    Handle loading/rescaling for the 4 tags (and loader state), using DCP.

    Args:
        loader: StatefulDataLoader instance to load into
        path: Directory path to load checkpoint from
        device_mesh: Device mesh for distributed tensors
    """
    base = loader.state_dict()
    nworkers = base["_snapshot"]["_main_snapshot"]["_num_workers"]
    # TODO: do we have to get rank and worldsize from loader.dataset? make it get from loader instead?
       # can get this from the loader
    r = loader.dataset.rank
    w = loader.dataset.worldsize

    # Extract current worker states structure
    dstate = _extract_worker_states(base, nworkers)

    # Determine if we're rescaling
    ckp_ws, ckp_nw, easy_load = _check_rescaling(path, w, nworkers)

    # Get checkpoint metadata
    meta = get_checkpoint_metadata(path)

    # Load each category
    if easy_load:
        base, reshard_sizes = _load_state_vars_easy(
            dstate, meta, path, device_mesh, r, ckp_nw
        )
    else:
        reshard_sizes = None

    _load_broadcast_vars(dstate, meta, path, nworkers)
    _load_reshard_vars(
        dstate, meta, path, device_mesh, r, w, nworkers, easy_load, reshard_sizes
    )
    _load_custom_vars(dstate, meta, path, r, nworkers, ckp_ws, ckp_nw, easy_load)

    # Reconstruct loader state dict and load
    _finalize_and_load(loader, base, dstate, nworkers)


def _extract_worker_states(base: Dict, nworkers: int) -> Dict[str, List[Dict]]:
    """
    Extract and restructure worker dataset states from loader state dict.

    Args:
        base: Loader state dict
        nworkers: Number of workers

    Returns:
        Dictionary with keys {state, broadcast, reshard, custom}
    """
    dstate = base["_snapshot"]["_worker_snapshots"]
    dstate = [dstate[f"worker_{i}"].pop("dataset_state") for i in range(len(dstate))]
    # Flip List[dict[dict]] to dict[List[dict]]
    dstate = {k: [d[k] for d in dstate] for k in dstate[0].keys()}
    return dstate


def _check_rescaling(path: str, worldsize: int, nworkers: int) -> Tuple[int, int, bool]:
    """
    Determine if we're rescaling based on checkpoint metadata.

    Args:
        path: Checkpoint path
        worldsize: Current worldsize
        nworkers: Current number of workers

    Returns:
        Tuple of (checkpoint_worldsize, checkpoint_nworkers, is_easy_load)
    """
    ckp_ws = (
        0
        if not os.path.exists(path)
        else len([x for x in os.listdir(path) if ".distcp" in x])
    )
    d = {"broadcast": {"global_worldsize": 0}}
    checkpoint.load(
        state_dict=d,
        storage_reader=checkpoint.FileSystemReader(path=path),
    )
    ckp_nw = d["broadcast"]["global_worldsize"] // ckp_ws
    easy_load = ckp_ws == worldsize and ckp_nw == nworkers
    return ckp_ws, ckp_nw, easy_load


def _load_state_vars_easy(
    dstate: Dict,
    meta: Dict,
    path: str,
    device_mesh: dist.DeviceMesh,
    rank: int,
    ckp_nw: int,
) -> Tuple[Dict, Dict]:
    """
    Load state variables when not rescaling.

    Args:
        dstate: Worker states dict (modified in place)
        meta: Checkpoint metadata
        path: Checkpoint directory path
        device_mesh: Device mesh
        rank: Current rank
        ckp_nw: Checkpoint number of workers

    Returns:
        Tuple of (loader_base_state, reshard_sizes)
    """
    # Build placeholder DTensors, load straightforwardly
    state_vars = crawl(
        {},
        meta["state"],
        functools.partial(build_dtensor, rank=rank, mesh=device_mesh),
    )
    checkpoint.load(
        state_dict={"state": state_vars},
        storage_reader=checkpoint.FileSystemReader(path=path),
    )
    # Convert back from dtensor
    state_vars = crawl(state_vars, meta["state"], unwrap_dtensor)

    # Pull out manually added subdicts
    base = state_vars.pop("loader_state")
    reshard_sizes = state_vars.pop("reshard_sizes")

    # Flip dict[List] to List[dict]
    state_vars = [{k: state_vars[k][i] for k in state_vars} for i in range(ckp_nw)]
    dstate["state"] = state_vars

    return base, reshard_sizes


def _load_broadcast_vars(dstate: Dict, meta: Dict, path: str, nworkers: int) -> None:
    """
    Load broadcast variables and replicate across workers.

    Args:
        dstate: Worker states dict (modified in place)
        meta: Checkpoint metadata
        path: Checkpoint directory path
        nworkers: Number of workers
    """
    broadcast_vars = {k: None for k in meta["broadcast"]}
    checkpoint.load(
        state_dict={"broadcast": broadcast_vars},
        storage_reader=checkpoint.FileSystemReader(path=path),
    )
    dstate["broadcast"] = [broadcast_vars] * nworkers


def _load_reshard_vars(
    dstate: Dict,
    meta: Dict,
    path: str,
    device_mesh: dist.DeviceMesh,
    rank: int,
    worldsize: int,
    nworkers: int,
    easy_load: bool,
    reshard_sizes: Dict,
) -> None:
    """
    Load reshard variables, handling both easy load and rescaling cases.

    Args:
        dstate: Worker states dict (modified in place)
        meta: Checkpoint metadata
        path: Checkpoint directory path
        device_mesh: Device mesh
        rank: Current rank
        worldsize: Current worldsize
        nworkers: Number of workers
        easy_load: Whether we're doing easy load (no rescaling)
        reshard_sizes: Per-worker shard sizes (from state, for easy load)
    """
    if easy_load:
        reshard_vars, local_split = _build_reshard_easy(
            meta, device_mesh, rank, reshard_sizes
        )
    else:
        reshard_vars, local_split = _build_reshard_rescale(
            meta, device_mesh, rank, worldsize, nworkers
        )

    checkpoint.load(
        state_dict={"reshard": reshard_vars},
        storage_reader=checkpoint.FileSystemReader(path=path),
    )

    # Convert from dtensor back to List[tensor]
    # After DCP loads the rank's DTensor, split it back into per-worker tensors
    reshard_vars = {
        k: v.to_local().local_shards()[0].split(local_split[k])
        for k, v in reshard_vars.items()
    }

    # Flip dict[List] to List[dict]
    dstate["reshard"] = [
        {k: v[i] for k, v in reshard_vars.items()} for i in range(nworkers)
    ]


def _build_reshard_easy(
    meta: Dict,
    device_mesh: dist.DeviceMesh,
    rank: int,
    reshard_sizes: Dict,
) -> Tuple[Dict, Dict]:
    """
    Build reshard DTensors for easy load case (no rescaling).

    Reconstructs LocalShardsWrappers from corresponding ChunkMetadata.

    Args:
        meta: Checkpoint metadata
        device_mesh: Device mesh
        rank: Current rank
        reshard_sizes: Per-worker shard sizes from checkpoint

    Returns:
        Tuple of (reshard_vars dict, local_split dict)
    """
    reshard_vars = {
        k: dtensor.DTensor.from_local(
            local_tensor=LocalShardsWrapper(
                local_shards=[
                    torch.empty(
                        v.chunks[rank].sizes,
                        dtype=v.properties.dtype,
                    )
                ],
                local_offsets=[v.chunks[rank].offsets],
            ),
            device_mesh=device_mesh,
            placements=[dtensor.placement_types.Shard(0)],
            shape=v.size,
            stride=tuple([1] * len(v.size)),
        )
        for k, v in meta["reshard"].items()
    }
    local_split = reshard_sizes
    return reshard_vars, local_split


def _build_reshard_rescale(
    meta: Dict,
    device_mesh: dist.DeviceMesh,
    rank: int,
    worldsize: int,
    nworkers: int,
) -> Tuple[Dict, Dict]:
    """
    Build reshard DTensors for rescaling case.

    Uses global size to construct resharded LocalShardsWrappers.

    Args:
        meta: Checkpoint metadata
        device_mesh: Device mesh
        rank: Current rank
        worldsize: New worldsize
        nworkers: Number of workers

    Returns:
        Tuple of (reshard_vars dict, local_split dict)
    """
    reshard_vars = {}
    local_split = {}

    for k, v in meta["reshard"].items():
        # Calculate this rank's slice boundaries from the global tensor
        rank_offsets = [(i * v.size[0]) // worldsize for i in range(worldsize)] + [
            v.size[0]
        ]
        local_shard_offset = rank_offsets[rank]
        local_shard_shape = [rank_offsets[rank + 1] - local_shard_offset] + list(
            v.size[1:]
        )
        local_shard_offsets = [local_shard_offset] + [0] * (len(local_shard_shape) - 1)

        reshard_vars[k] = dtensor.DTensor.from_local(
            local_tensor=LocalShardsWrapper(
                local_shards=[
                    torch.empty(
                        local_shard_shape,
                        dtype=v.properties.dtype,
                    )
                ],
                local_offsets=[local_shard_offsets],
            ),
            device_mesh=device_mesh,
            placements=[dtensor.placement_types.Shard(0)],
            shape=v.size,
            stride=tuple([1] * len(v.size)),
        )

        # Further split this rank's shard across workers
        local_split[k] = [
            (i * local_shard_shape[0]) // nworkers for i in range(nworkers)
        ] + [local_shard_shape[0]]
        local_split[k] = [
            local_split[k][i + 1] - local_split[k][i] for i in range(nworkers)
        ]

    return reshard_vars, local_split


def _load_custom_vars(
    dstate: Dict,
    meta: Dict,
    path: str,
    rank: int,
    nworkers: int,
    ckp_ws: int,
    ckp_nw: int,
    easy_load: bool,
) -> None:
    """
    Load custom variables, handling both easy load and rescaling cases.

    Args:
        dstate: Worker states dict (modified in place)
        meta: Checkpoint metadata
        path: Checkpoint directory path
        rank: Current rank
        nworkers: Number of workers
        ckp_ws: Checkpoint worldsize
        ckp_nw: Checkpoint number of workers
        easy_load: Whether we're doing easy load
    """
    if easy_load:
        _load_custom_easy(dstate, meta, path, rank, nworkers)
    else:
        _load_custom_rescale(dstate, meta, path, ckp_ws, ckp_nw, nworkers)


def _load_custom_easy(
    dstate: Dict, meta: Dict, path: str, rank: int, nworkers: int
) -> None:
    """
    Load custom variables when not rescaling.

    Only loads the current rank's keys.

    Args:
        dstate: Worker states dict (modified in place)
        meta: Checkpoint metadata
        path: Checkpoint directory path
        rank: Current rank
        nworkers: Number of workers
    """
    prefixes = [f"rank{rank * nworkers + i}" for i in range(nworkers)]
    custom_vars = {
        k: (
            torch.empty(v.size, dtype=v.properties.dtype)
            if isinstance(v, TensorStorageMetadata)
            else None
        )
        for k, v in meta["custom"].items()
        if k[: k.find(".")] in prefixes
    }
    checkpoint.load(
        state_dict={"custom": custom_vars},
        storage_reader=checkpoint.FileSystemReader(path=path),
    )
    # Convert dict of rank.[keys] to list[dict]
    custom_vars = [
        {
            k[k.find(".") + 1 :]: v
            for k, v in custom_vars.items()
            if k[: k.find(".")] == p
        }
        for p in prefixes
    ]
    print(".   ", custom_vars)
    dstate["custom"] = custom_vars


def _load_custom_rescale(
    dstate: Dict,
    meta: Dict,
    path: str,
    ckp_ws: int,
    ckp_nw: int,
    nworkers: int,
) -> None:
    """
    Load custom variables when rescaling.

    Loads all ranks' keys and compiles into global list for custom resharding.

    Args:
        dstate: Worker states dict (modified in place)
        meta: Checkpoint metadata
        path: Checkpoint directory path
        ckp_ws: Checkpoint worldsize
        ckp_nw: Checkpoint number of workers
        nworkers: Current number of workers
    """
    custom_vars = {
        k: (
            torch.empty(v.size, dtype=v.properties.dtype)
            if isinstance(v, TensorStorageMetadata)
            else None
        )
        for k, v in meta["custom"].items()
    }
    checkpoint.load(
        state_dict={"custom": custom_vars},
        storage_reader=checkpoint.FileSystemReader(path=path),
    )

    # Convert dict of rank.[keys] to List[dict] by pulling out rank prefixes
    custom_vars = [
        {
            k[k.find(".") + 1 :]: v
            for k, v in custom_vars.items()
            if f"rank{i}" == k[: len(f"rank{i}")]
        }
        for i in range(ckp_ws * ckp_nw)
    ]

    # Flip list[dict] into dict[list]
    custom_vars = {k: [d[k] for d in custom_vars] for k in custom_vars[0]}

    # Set __rescaling__ True
    custom_vars["__rescaling__"] = True
    dstate["custom"] = [custom_vars] * nworkers


def _finalize_and_load(
    loader: Any,
    base: Dict,
    dstate: Dict,
    nworkers: int,
) -> None:
    """
    Reconstruct loader state dict and load into loader.

    Args:
        loader: StatefulDataLoader to load into
        base: Base loader state dict
        dstate: Processed worker states
        nworkers: Number of workers
    """
    # Flip dict[list[dict]] into list[dict[dict]]
    dstate = [{k: dstate[k][i] for k in dstate} for i in range(nworkers)]

    # Load worker dstates back into loader's dataset_state
    for i in range(nworkers):
        base["_snapshot"]["_worker_snapshots"][f"worker_{i}"]["dataset_state"] = dstate[
            i
        ]

    loader.load_state_dict(base)
