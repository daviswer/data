import functools
import os
from copy import deepcopy
from typing import cast, Optional, Union

import torch
from torch.distributed import checkpoint
import torch.distributed.tensor as dtensor
import torch.distributed as dist
from torch.distributed.tensor._shards_wrapper import LocalShardsWrapper
from torch.distributed.checkpoint._storage_utils import _storage_setup
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.storage import StorageReader


from ..stateful_dataloader.stateful_dataloader import StatefulDataLoader


"""
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


def save_ckpt_dcp(
    loader: StatefulDataLoader,
    path: str,
    device_mesh: dist.DeviceMesh,
):
    """
    Retrieves dataloader state dict, and separates worker states from loader state.
    Aggregates worker states, and wraps/processes state/broadcast/reshard/custom variables.
    Saves states concurrently through DCP APIs. 
    """
    os.makedirs(path, exist_ok=True)
    rank = loader.dataset.rank
    worldsize = loader.dataset.worldsize
    state = deepcopy(loader.state_dict())
    nworkers = state["_snapshot"]["_main_snapshot"]["_num_workers"]
    dstate = state["_snapshot"]["_worker_snapshots"]
    dstate = [dstate[f"worker_{i}"].pop("dataset_state") for i in range(len(dstate))]  # List[dict]
    # Flip List[dict[dict]] to dict[List[dict]]
    dstate = {k:[d[k] for d in dstate] for k in dstate[0].keys()}  # {state, broadcast, reshard, custom}

    # State dict: add loader state, wrap in DTensor (no special wrapper)
    # We don't care about rescaling behavior here because it'll get dropped in that case
    state_vars = dstate["state"]
    # Flip List[dict] to dict[List]
    state_vars = {k:[d[k] for d in state_vars] for k in state_vars[0].keys()}
    state_vars["loader_state"] = state
    # Pause until reshard can add its contribution

    # Broadcast dict: save only first entry, only if rank is 0
    broadcast_vars = dstate.pop("broadcast")[0]
    if rank == 0:
        dstate["broadcast"] = broadcast_vars
        # Add ckpt worldsize
        dstate["broadcast"]["global_worldsize"] = worldsize * nworkers

    # Reshard dict: concatenate entries, fetch global size, 
    # wrap in sharding DTensor, store partial sizes in State
    reshard_vars = dstate["reshard"]
    # Assert all reshard vals are tensors
    for k,v in reshard_vars[0].items():
        assert isinstance(v, torch.Tensor), f"Reshard var {k} is not a torch tensor!"
    # Flip list[dict] to dict[list]
    reshard_vars = {k:[d[k] for d in reshard_vars] for k in reshard_vars[0].keys()}
    # Inject per-worker shard sizes into dstate["state"] for use when not rescaling
    state_vars["reshard_sizes"] = {
        k:[x.size(0) for x in v]
        for k,v in reshard_vars.items()
    }
    # Concat and wrap in DTensor, fetching global sizes
    def wrap_shardtensor(x, mesh):
        size = torch.tensor(x.size(0), dtype=torch.long)[None]
        sizes = torch.empty(worldsize, dtype=torch.long)
        dist.all_gather_into_tensor(sizes, size,group=mesh.get_group(0))
        offsets = sizes.cumsum(0).roll(1, 0)
        offsets[0] = 0
        global_shape = [sizes.sum()] + list(x.shape[1:])
        x = LocalShardsWrapper(
            local_shards=[x], local_offsets=[(offsets[rank], 0)]
        )
        x = dtensor.DTensor.from_local(
            local_tensor=x,
            device_mesh=mesh,
            placements=[dtensor.placement_types.Shard(0)],
            shape=torch.Size(global_shape),
            stride=torch.Size([1]*len(global_shape)),
        )
        return x
    reshard_vars = {
        k:wrap_shardtensor(torch.cat(v, dim=0), device_mesh)
        for k,v in reshard_vars.items()
    }
    dstate["reshard"] = reshard_vars

    # Finish up state now that reshard has added its size metadata
    def wrap_dtensor(d, mesh):
        for k,v in d.items():
            if isinstance(v, dict):
                d[k] = wrap_dtensor(v, mesh)
            else:
                assert not isinstance(v, torch.Tensor), f"DCP saving does not currently support tensor state values. Please convert state var {k} to (nested) list."
                if v is None:
                    v = float("inf")
                v = torch.tensor(v)[None]
                d[k] = dtensor.DTensor.from_local(v, mesh, [dtensor.placement_types.Shard(0)])
        return d
    state_vars = wrap_dtensor(state_vars, device_mesh)
    dstate["state"] = state_vars

    # Custom: prepend rank to every key
    custom_vars = dstate["custom"]
    # Convert list[dict] to dict with prepended rank in keys
    custom_vars = {f"rank{rank*nworkers+i}."+k : custom_vars[i][k] for i in range(len(custom_vars)) for k in custom_vars[0].keys()}
    dstate["custom"] = custom_vars

    checkpoint.save(
        dstate,
        storage_writer=checkpoint.FileSystemWriter(path=path), 
        planner = checkpoint.DefaultSavePlanner(),
    )


def load_ckpt_dcp(
    loader: StatefulDataLoader,
    path: str,
    device_mesh: dist.DeviceMesh,
):
    """
    Retrieves dataloader state dict, and separates worker states from loader state.
    Handle loading/rescaling for the 4 tags (and loader state), using DCP.
    """
    base = loader.state_dict()
    nworkers = base["_snapshot"]["_main_snapshot"]["_num_workers"]
    r = loader.dataset.rank
    w = loader.dataset.worldsize
    dstate = base["_snapshot"]["_worker_snapshots"]
    dstate = [dstate[f"worker_{i}"].pop("dataset_state") for i in range(len(dstate))]  # List[dict]
    # Flip List[dict[dict]] to dict[List[dict]]
    dstate = {k:[d[k] for d in dstate] for k in dstate[0].keys()}  # {state, broadcast, reshard, custom}    inp = {"state":deepcopy(base), "dstate":dstate}
    
    # Determine if we're rescaling or not
    ckp_ws = 0 if not os.path.exists(path) else len([x for x in os.listdir(path) if ".distcp" in x])
    d = {'broadcast':{'global_worldsize':0}}
    checkpoint.load(
        state_dict = d,
        storage_reader = checkpoint.FileSystemReader(path=path),
    )
    ckp_nw = d['broadcast']['global_worldsize'] // ckp_ws
    easy_load = ckp_ws == w and ckp_nw == nworkers
    
    # Fetch checkpoint metadata
    def list_stored_state_dict(
        checkpoint_id: Union[str, os.PathLike, None] = None,
        storage_reader: Optional[StorageReader] = None,
    ):
        """
        Copied from https://github.com/pytorch/pytorch/pull/160610
        List the stored checkpoint metadata.
        NB: The returned state-dict keys are flattened.
        """
        storage_reader = cast(
            StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
        )
        md = storage_reader.read_metadata()
        sd = md.state_dict_metadata  # flattened dict.
        return sd
    meta_flat = list_stored_state_dict(checkpoint_id=path)
    # Unflatten dict one level
    meta = {field:{k[len(field)+1:]:v for k,v in meta_flat.items() if field in k[:k.find('.')]} 
            for field in ["state","broadcast","reshard","custom"]}
    # Unflatten reshard sizes
    loaderflags = [k[k.find('.')+1:] for k in meta["state"] if "reshard_sizes" in k[:k.find('.')]]
    meta["state"]["reshard_sizes"] = {k:meta["state"].pop("reshard_sizes."+k) for k in loaderflags}
    # Unflatten loader state fully. Avoid unflattening other state subdicts.
    loaderflags = [k[k.find('.')+1:] for k in meta["state"] if "loader_state" in k[:k.find('.')]]
    loadermeta = {}
    for key in loaderflags:
        trace = key.split('.')
        d = loadermeta
        for subk in trace[:-1]:
            if subk not in d:
                d[subk] = {}
            d = d[subk]
        d[trace[-1]] = meta["state"].pop("loader_state."+key)
    meta["state"]["loader_state"] = loadermeta
    
    # State: load if easy, otherwise ignore
    if easy_load:
        def crawl(d, m, f):
        # Crawl nested dict d using metadata m, applying function f to every non-dict entry.
        # If d doesn't have a corresponding entry from m, creates one.
        # Function f should take two arguments, the entry from d and from m respectively.
            for k,v in m.items():
                if isinstance(v, dict):
                    d[k] = crawl(d.get(k, {}), m[k], f)
                else:
                    d[k] = f(d.get(k, None), v)
            return d
        def build_dtensor(x, meta, rank, mesh):
            x = torch.empty(meta.chunks[rank].sizes, dtype=meta.properties.dtype)
            return dtensor.DTensor.from_local(x, mesh, [dtensor.placement_types.Shard(0)])
        # Built placeholder DTensors, load straightforwardly
        state_vars = crawl({}, meta['state'], functools.partial(build_dtensor, rank=r, mesh=device_mesh))
        checkpoint.load(
            state_dict = {"state":state_vars},
            storage_reader = checkpoint.FileSystemReader(path=path),
        )
        # Convert back from dtensor
        def unwrap_dtensor(x, _):
            x = x.to_local().tolist()[0]
            if x == float("inf"):
                x = None
            return x
        state_vars = crawl(state_vars, meta['state'], unwrap_dtensor)
        # Pull out manually added subdicts
        base = state_vars.pop("loader_state")
        reshard_sizes = state_vars.pop("reshard_sizes")
        # Flip dict[List] to List[dict]
        state_vars = [{k:state_vars[k][i] for k in state_vars} for i in range(ckp_nw)]
        dstate["state"] = state_vars

    # Broadcast: load relevant key subset, replicate across workers
    broadcast_vars = {k:None for k in meta['broadcast']}  # Assuming flat dict
    checkpoint.load(
        state_dict = {"broadcast":broadcast_vars},
        storage_reader = checkpoint.FileSystemReader(path=path),
    )
    dstate["broadcast"] = [broadcast_vars] * nworkers
    
    # Reshard: build local plans from metadata
    if easy_load:
        # Load back individual mismatched shards by reconstructing LocalShardsWrappers 
        # from corresponding ChunkMetadata
        reshard_vars = {
            k: dtensor.DTensor.from_local(
                local_tensor = LocalShardsWrapper(
                    local_shards=[torch.empty(
                        v.chunks[r].sizes, 
                        dtype=v.properties.dtype,
                    )], 
                    local_offsets=[v.chunks[r].offsets]
                ),
                device_mesh = device_mesh,
                placements = [dtensor.placement_types.Shard(0)],
                shape = v.size,
                stride = tuple([1] * len(v.size)),
            ) for k,v in meta['reshard'].items()
        }
        # Retrieve local worker splits from "state"
        local_split = reshard_sizes
    else:
        reshard_vars = {}
        local_split = {}
        # Use global size to construct resharded LocalShardsWrappers
        for k,v in meta["reshard"].items():
            offsets = [(i*v.size[0])//w for i in range(w)] + [v.size[0]]
            my_offset = offsets[r]
            my_size = [offsets[r+1] - my_offset] + list(v.size[1:])
            my_offset = [my_offset] + [0]*(len(my_size)-1)
            reshard_vars[k] = dtensor.DTensor.from_local(
                local_tensor = LocalShardsWrapper(
                    local_shards=[torch.empty(
                        my_size,
                        dtype = v.properties.dtype,
                    )],
                    local_offsets=[my_offset]
                ),
                device_mesh = device_mesh,
                placements = [dtensor.placement_types.Shard(0)],
                shape = v.size,
                stride = tuple([1] * len(v.size)),
            )
            # Repeat sharding process for local partition over workers
            local_split[k] = [(i*my_size[0])//nworkers for i in range(nworkers)] + [my_size[0]]
            local_split[k] = [local_split[k][i+1]-local_split[k][i] for i in range(nworkers)]
    checkpoint.load(
        state_dict={"reshard": reshard_vars},
        storage_reader=checkpoint.FileSystemReader(path=path),
    )
    # Convert from dtensor back to List[tensor]
    reshard_vars = {
        k: v.to_local().local_shards()[0].split(local_split[k])
        for k,v in reshard_vars.items()
    }  # Assuming flat dict
    # Flip dict[List] to List[dict]
    dstate["reshard"] = [{k:v[i] for k,v in reshard_vars.items()} for i in range(nworkers)]

    # Custom: load individually or in aggregate, based on key rank-prefixes
    if easy_load:
        # Load only the current rank's key(s)
        prefixes = [f"rank{r*nworkers+i}" for i in range(nworkers)]
        custom_vars = {
            k : torch.empty(v.size, dtype=v.properties.dtype) if isinstance(v, TensorStorageMetadata) else None 
            for k,v in meta["custom"].items() if k[:k.find(".")] in prefixes
        }
        checkpoint.load(
            state_dict={"custom": custom_vars},
            storage_reader=checkpoint.FileSystemReader(path=path),
        )
        # Convert dict of rank.[keys] to list[dict]
        custom_vars = [{k[k.find(".")+1:]:v for k,v in custom_vars.items() if k[:k.find(".")] == p} for p in prefixes]
        dstate["custom"] = custom_vars
    else:
        # Load keys across ranks, compile each rankset into global list. 
        # Pop and reset __rescaling__ flag.
        custom_vars = {
            k : torch.empty(v.size, dtype=v.properties.dtype) if isinstance(v, TensorStorageMetadata) else None 
            for k,v in meta["custom"].items()
        }
        checkpoint.load(
            state_dict={"custom": custom_vars},
            storage_reader=checkpoint.FileSystemReader(path=path),
        )
        # Convert dict of rank.[keys] to List[dict] by pulling out rank prefixes
        custom_vars = [
            {
                k[k.find(".")+1:] : v
                for k,v in custom_vars.items()
                if f"rank{i}" == k[:len(f"rank{i}")]
            } for i in range(ckp_ws * ckp_nw)
        ]
        # Flip list[dict] into dict[list]
        custom_vars = {k:[d[k] for d in custom_vars] for k in custom_vars[0]}
        # Set __rescaling__ True
        custom_vars["__rescaling__"] = True
        dstate["custom"] = [custom_vars] * nworkers

    # Flip dict[list[dict]] into list[dict[dict]]
    dstate = [{k:dstate[k][i] for k in dstate} for i in range(nworkers)]
    # Load worker dstates back into loader
    for i in range(nworkers):
        base["_snapshot"]["_worker_snapshots"][f"worker_{i}"]["dataset_state"] = dstate[i]
    loader.load_state_dict(base)