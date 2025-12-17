"""
Checkpoint metadata utilities for DCP.

Provides helpers for reading and manipulating checkpoint metadata,
including flattened key extraction and unflattening.
"""

import os
from typing import cast, Dict, Optional, Union

from torch.distributed.checkpoint._storage_utils import _storage_setup
from torch.distributed.checkpoint.storage import StorageReader


def list_stored_state_dict(
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_reader: Optional[StorageReader] = None,
) -> Dict:
    """
    List the stored checkpoint metadata.

    Copied from https://github.com/pytorch/pytorch/pull/160610
    NB: The returned state-dict keys are flattened.

    Args:
        checkpoint_id: Path to checkpoint directory
        storage_reader: Optional custom storage reader

    Returns:
        Flattened dictionary of metadata for all state dict entries
    """
    storage_reader = cast(
        StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
    )
    md = storage_reader.read_metadata()
    sd = md.state_dict_metadata  # flattened dict
    return sd


def unflatten_metadata(meta_flat: Dict) -> Dict:
    """
    Unflatten checkpoint metadata one level.

    Takes a flattened metadata dict with keys like "state.key1", "broadcast.key2"
    and returns a nested dict with top-level keys "state", "broadcast", etc.

    Args:
        meta_flat: Flattened metadata dictionary

    Returns:
        Nested dictionary with one level of unflattening
    """
    meta = {
        field: {
            k[len(field) + 1 :]: v
            for k, v in meta_flat.items()
            if field in k[: k.find(".")]
        }
        for field in ["state", "broadcast", "reshard", "custom"]
    }
    return meta


def unflatten_reshard_sizes(meta: Dict) -> Dict:
    """
    Unflatten the reshard_sizes subdict from state metadata.

    Args:
        meta: Partially unflattened metadata dict

    Returns:
        Same dict with reshard_sizes properly unflattened
    """
    loaderflags = [
        k[k.find(".") + 1 :]
        for k in meta["state"]
        if "reshard_sizes" in k[: k.find(".")]
    ]
    meta["state"]["reshard_sizes"] = {
        k: meta["state"].pop("reshard_sizes." + k) for k in loaderflags
    }
    return meta


def unflatten_loader_state(meta: Dict) -> Dict:
    """
    Fully unflatten the loader_state subdict from state metadata.

    The loader_state is deeply nested, so we need to reconstruct
    the full hierarchy from the flattened keys.

    Args:
        meta: Partially unflattened metadata dict

    Returns:
        Same dict with loader_state properly unflattened
    """
    loaderflags = [
        k[k.find(".") + 1 :]
        for k in meta["state"]
        if "loader_state" in k[: k.find(".")]
    ]
    loadermeta = {}
    for key in loaderflags:
        trace = key.split(".")
        d = loadermeta
        for subk in trace[:-1]:
            if subk not in d:
                d[subk] = {}
            d = d[subk]
        d[trace[-1]] = meta["state"].pop("loader_state." + key)
    meta["state"]["loader_state"] = loadermeta
    return meta


def get_checkpoint_metadata(path: str) -> Dict:
    """
    Read and process checkpoint metadata from disk.

    Combines list_stored_state_dict with unflattening operations
    to produce a usable metadata structure.

    Args:
        path: Path to checkpoint directory

    Returns:
        Fully processed metadata dictionary
    """
    meta_flat = list_stored_state_dict(checkpoint_id=path)
    meta = unflatten_metadata(meta_flat)
    meta = unflatten_reshard_sizes(meta)
    meta = unflatten_loader_state(meta)
    return meta
