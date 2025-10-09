import argparse
import math
import os
import pyarrow as pa
import time
import torch
from torch import distributed as dist
from copy import deepcopy

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.scalable_reader import (
    ArrowHandler,
    PreprocessDataset,
    DocPackingDataset,
    ScalableReader,
    save_ckpt_dcp,
)

parser = argparse.ArgumentParser(description="Script to validate rescaling of dataloader checkpoints")
parser.add_argument("--ckpt_path", type=str, default="./rescale_test")
parser.add_argument(
    "--logical_shards",
    type=int,
    default=20,
    help="Total number of data partitions. Must exceed (worldsize * n_workers) but not n_docs (1000).",
)
parser.add_argument("--seq_len", type=int, default=32, help="Batch seq len")
parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader workers per device")
parser.add_argument("--b_size", type=int, default=2, help="Number of data points per step per device")
parser.add_argument("--n_steps", type=int, default=30, help="Number of steps to take before saving. (n_steps * b_size * worldsize) cannot exceed number of items in epoch (3000)")
parser.add_argument("--n_bins", type=int, default=4, help="Number of packing/slicing bins")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()


# Setup
rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
dist.init_process_group(backend="gloo")
mesh = dist.device_mesh.init_device_mesh("cpu", [world_size])

# Check input args
assert args.logical_shards >= world_size*args.num_workers, f"Logical shards {args.logical_shards} cannot be less than total workers {world_size*args.num_workers}"
assert args.logical_shards <= 1000, f"Logical shards {args.logical_shards} cannot exceed number of documents 1000"
assert args.n_steps*args.b_size*world_size < 3000, f"Number of items drawn before saving {args.n_steps*args.b_size*world_size} cannot exceed number of document chunks 3000."

# Build dataset
datapath = os.path.join(args.ckpt_path, "dataset")
if not os.path.exists(datapath):
    if rank == 0:
        os.makedirs(datapath)
        schema = pa.schema([pa.field("tokens", pa.uint32())])
        with pa.ipc.new_file(
            os.path.join(datapath, "fileshard_1.arrow"), schema
        ) as writer:
            for i in range(500):
                out = list(range(i * 100, i * 100 + 100))
                writer.write(pa.record_batch([out], schema=schema))
        os.makedirs(os.path.join(datapath, "subfolder"))
        with pa.ipc.new_file(
            os.path.join(datapath, "subfolder/fileshard_2.arrow"), schema
        ) as writer:
            for i in range(500):
                out = list(range(50000 + i * 100, 50000 + i * 100 + 100))
                writer.write(pa.record_batch([out], schema=schema))
    else:
        # Give other ranks time for worker 0 to finish
        time.sleep(5)

# Build dataloader
data = ScalableReader(datapath, rank, world_size, ArrowHandler, -1, seed=args.seed, max_chunksize=40, n_logical_shards=args.logical_shards)
# Packing and slicing
data = DocPackingDataset(data, args.seq_len, 4, -1, -2, args.n_bins)
# Statelessly convert all outputs to tensors
data = PreprocessDataset(data, torch.tensor)
# Wrap in StatefulDataLoader
data = StatefulDataLoader(data, batch_size=args.b_size, num_workers=args.num_workers)

# If checkpoint does not exist, create it
ckpt_path = os.path.join(args.ckpt_path, "loader_dcp_state")
if not os.path.exists(ckpt_path) or len(os.listdir(ckpt_path)) == 0:
    if rank==0:
        print(f"Error: checkpoint {ckpt_path} does not exist!")
else:
    # state = deepcopy(data.state_dict())
    # dstate = state["_snapshot"]["_worker_snapshots"]
    # dstate = [dstate[f"worker_{i}"].pop("dataset_state") for i in range(len(dstate))]  # List[dict]
    # # Flip List[dict[dict]] to dict[List[dict]]
    # dstate = {k:[d[k] for d in dstate] for k in dstate[0].keys()}  # {state, broadcast, reshard, custom}
    # # Flip dict[List[dict]] to [dict[dict[List]]] and truncate
    # for k in dstate:
    #     dstate[k] = {k2:[d[k2] for d in dstate[k]][0] for k2 in dstate[k][0]}
    # # Pop custom subdict
    # dstate.pop('custom')

    # time.sleep(rank)
    # print(dstate)

    dstate = {'broadcast':{'global_worldsize':0}}
    dstate['state'] = {'loader_state':{'_snapshot':{'_snapshot_step':0}}}

    dist.checkpoint.load(
        dstate,
        storage_reader=dist.checkpoint.FileSystemReader(path=ckpt_path)
    )
    print()
    time.sleep(rank)
    print(dstate)