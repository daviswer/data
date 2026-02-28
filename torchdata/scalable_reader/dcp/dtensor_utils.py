"""
DTensor utilities for wrapping and unwrapping tensors for DCP checkpointing.

Provides helper functions for:
- Wrapping tensors in DTensors with sharding for distributed saving
- Unwrapping DTensors back to local tensors after loading
- Handling nested dictionaries of tensor/non-tensor values
"""

import torch
import torch.distributed as dist
import torch.distributed.tensor as dtensor
from torch.distributed.tensor._shards_wrapper import LocalShardsWrapper


def wrap_shardtensor(
    x: torch.Tensor,
    mesh: dist.DeviceMesh,
    rank: int,
    worldsize: int,
) -> dtensor.DTensor:
    """
    Wrap a local tensor in a sharded DTensor for distributed checkpointing.

    Gathers global sizes across all ranks and creates a DTensor with Shard(0)
    placement to enable DCP's distributed save/load.

    Args:
        x: Local tensor shard to wrap
        mesh: Device mesh for distribution
        rank: Current worker's rank
        worldsize: Total number of workers

    Returns:
        DTensor wrapping the local shard with global metadata
    """
    size = torch.tensor(x.size(0), dtype=torch.long)[None]
    sizes = torch.empty(worldsize, dtype=torch.long)
    dist.all_gather_into_tensor(sizes, size, group=mesh.get_group("dp"))
    offsets = sizes.cumsum(0).roll(1, 0)
    offsets[0] = 0
    global_shape = [sizes.sum()] + list(x.shape[1:])
    x = LocalShardsWrapper(local_shards=[x], local_offsets=[(offsets[rank], 0)])
    x = dtensor.DTensor.from_local(
        local_tensor=x,
        device_mesh=mesh,
        placements=[dtensor.placement_types.Shard(0)],
        shape=torch.Size(global_shape),
        stride=torch.Size([1] * len(global_shape)),
    )
    return x


def wrap_dtensor(d: dict, mesh: dist.DeviceMesh) -> dict:
    """
    Recursively wrap scalar values in a nested dict as DTensors.

    Used for state variables that need to be saved via DCP. Each scalar
    value is converted to a 1D tensor and wrapped as a sharded DTensor.

    Args:
        d: Dictionary (possibly nested) of state values
        mesh: Device mesh for distribution

    Returns:
        Same structure with values wrapped as DTensors
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = wrap_dtensor(v, mesh)
        else:
            assert not isinstance(
                v, torch.Tensor
            ), f"DCP saving does not currently support tensor state values. Please convert state var {k} to (nested) list."
            if v is None:
                v = float("inf")
            v = torch.tensor(v)[None]
            d[k] = dtensor.DTensor.from_local(
                v, mesh, [dtensor.placement_types.Shard(0)]
            )
    return d


def unwrap_dtensor(x, _) -> any:
    """
    Unwrap a DTensor back to its original scalar value.

    Converts the DTensor to a local tensor and extracts the scalar.
    Handles the special case of infinity (used as a sentinel for None).

    Args:
        x: DTensor to unwrap
        _: Unused (for compatibility with crawl function signature)

    Returns:
        Original scalar value
    """
    x = x.to_local().tolist()[0]
    if x == float("inf"):
        x = None
    return x


def build_dtensor(
    x, meta, rank: int, mesh: dist.DeviceMesh
) -> dtensor.DTensor:
    """
    Build an empty DTensor placeholder from checkpoint metadata.

    Used during loading to create the correct shaped tensor before
    DCP populates it with actual data.

    Args:
        x: Unused (placeholder value)
        meta: TensorStorageMetadata with shape/dtype info
        rank: Current worker's rank
        mesh: Device mesh for distribution

    Returns:
        Empty DTensor with correct shape for this rank
    """
    x = torch.empty(meta.chunks[rank].sizes, dtype=meta.properties.dtype, device="cpu")
    return dtensor.DTensor.from_local(
        x, mesh, [dtensor.placement_types.Shard(0)]
    )


def crawl(d: dict, m: dict, f: callable) -> dict:
    """
    Crawl nested dict d using metadata m, applying function f to every non-dict entry.

    If d doesn't have a corresponding entry from m, creates one.
    Function f should take two arguments: the entry from d and from m respectively.

    Args:
        d: Dictionary to crawl/transform
        m: Metadata dictionary with same structure
        f: Function to apply to leaf values

    Returns:
        Transformed dictionary
    """
    for k, v in m.items():
        if isinstance(v, dict):
            d[k] = crawl(d.get(k, {}), m[k], f)
        else:
            d[k] = f(d.get(k, None), v)
    return d
