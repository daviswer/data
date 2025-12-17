"""
DCP checkpoint utilities for scalable dataloaders.

This module is maintained for backward compatibility.

The following saving/loading functions use PyTorch DCP to handle distributed transfer of dataloader
state dict objects to and from disk. Implements specified scaling behavior for the four tag types
of ScalableReader and _NestedStatefulDatasets, depending on whether or not rescaling is being performed
at load time:

1. State: State variables are wrapped in DTensors and saved via DCP. At load time, values are loaded
back only when not rescaling. If rescaling, the checkpoint values are ignored. Some additional metadata
saved under this category, including: 1) the state_dict of the torchdata DataLoader itself, and 2) the
sizes of the reshard variables across each rank's individual worker processes.
2. Broadcast: Broadcast variables are saved from rank 0 only as generic state dict entries. At load
time, these values are loaded back and replicated to each rank and worker.
3. Reshard: Reshard variables are wrapped in DTensors, sharding on dim 0, with added support for
different sizes in dim 0 across workers and ranks. When loading without rescaling, the (possibly
uneven) shards are loaded back exactly as saved. When rescaling, entries are pooled into a single tensor,
and resharded on dim 0 as evenly as possible across ranks, then workers.
4. Custom: Custom variables are saved as generic state dict entries, with global rank prepended to
keys to prevent dict collisions. When loading without rescaling, only the entry from the same rank
is loaded back. When rescaling, all ranks' worth of entries are loaded, and passed into the Dataset
as a state dict of lists of values. The _StatefulDataset.custom_fns are then used to perform custom
resharding as specified, during _StatefulDataset.load_state_dict().

This approach imposes the following restrictions on checkpoint format:

1. Top-level dict has four keys ("state", "broadcast", "reshard", "custom") holding subdicts for
each of the four tag categories above.
2. Subdicts (ignoring the additional metadata added to "state") are assumed to be flat. Thus any
additions to the _StatefulDataset pipeline must also maintain this "dict of flat subdicts" format.
3. Every value in "state" must be convertible into a torch.Tensor when arranged into a list.
In particular, if a variable in "state" has values x1, x2, x3 across 3 workers, then
torch.tensor([x1,x2,x3]) must produce a legal torch.Tensor. Note that this restriction also applies
to StatefulDataLoader state dict entries, since these are placed inside of the "state" subdict.
4. Every value in "broadcast" is assumed to NOT be a torch.Tensor (as DCP handles tensors and
non-tensors differently).
5. Every value in "reshard" must be a torch.Tensor, with resharding performed on dim 0.
6. Every value in "custom" must be EITHER a torch.Tensor, or a non-tensor or other data structure
containing only non-tensors. Behavior for list[torch.Tensor], for example,  is currently undefined
(due to DCP's separate handling of tensors vs non-tensors).

These can be addressed with further effort, if they prove problematic.
"""

# Re-export from modular implementation for backward compatibility
from .dcp import load_ckpt_dcp, save_ckpt_dcp

__all__ = ["save_ckpt_dcp", "load_ckpt_dcp"]
