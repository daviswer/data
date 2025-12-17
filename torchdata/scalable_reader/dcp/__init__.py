"""
DCP (Distributed Checkpoint) utilities for scalable dataloader checkpointing.

This module provides functions for saving and loading dataloader state dicts
using PyTorch's Distributed Checkpointing (DCP) framework, with support for
rescaling to different numbers of workers.

The module handles four types of state variables differently:

1. State: Scalar values wrapped in DTensors, dropped when rescaling
2. Broadcast: Values saved from rank 0 only, replicated on load
3. Reshard: Tensors sharded on dim 0, re-partitioned when rescaling
4. Custom: User-defined resharding via custom functions

Usage:
    from scalable_reader.dcp import save_ckpt_dcp, load_ckpt_dcp

    # Save checkpoint
    save_ckpt_dcp(loader, "/path/to/checkpoint", device_mesh)

    # Load checkpoint (handles rescaling automatically)
    load_ckpt_dcp(loader, "/path/to/checkpoint", device_mesh)
"""

from .load import load_ckpt_dcp
from .save import save_ckpt_dcp

__all__ = ["save_ckpt_dcp", "load_ckpt_dcp"]
