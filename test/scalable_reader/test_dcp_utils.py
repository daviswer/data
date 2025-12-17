"""
Shared utilities for DCP tests.

This module provides common test fixtures, data generation, and helper functions
used across DCP test files (test_dcp.py and test_dcp_rescale.py).
"""

import functools
import os
import tempfile

import pyarrow as pa
import torch
import torch.distributed as dist

from torchdata.scalable_reader import (
    ArrowHandler,
    DocPackingDataset,
    PreprocessDataset,
    SamplingDataset,
    ScalableReader,
    ShuffleDataset,
)


#### -------------------------    DATA GENERATION    ------------------------- ####


def generate_sequential_multidata():
    """
    Generates test data in a temp directory, and returns that tempdir object.
    Two dataset folders: one has a large shardfile (100x100), other has two small shardfiles (50x50)
    """
    tmpdir = tempfile.TemporaryDirectory()
    schema = pa.schema([pa.field("tokens", pa.uint32())])

    os.mkdir(os.path.join(tmpdir.name, "dataset_1"))
    os.mkdir(os.path.join(tmpdir.name, "dataset_2"))
    os.mkdir(os.path.join(tmpdir.name, "dataset_2", "subfolder"))

    with pa.ipc.new_file(
        os.path.join(tmpdir.name, "dataset_1/fullshard.arrow"), schema
    ) as writer:
        for i in range(100):
            out = list(range(i * 100, i * 100 + 99))
            writer.write(pa.record_batch([out], schema=schema))

    with pa.ipc.new_file(
        os.path.join(tmpdir.name, "dataset_2/quartershard_1.arrow"), schema
    ) as writer:
        for i in range(50):
            out = list(range(i * 50, i * 50 + 49))
            writer.write(pa.record_batch([out], schema=schema))

    with pa.ipc.new_file(
        os.path.join(tmpdir.name, "dataset_2/subfolder/quartershard_2.arrow"), schema
    ) as writer:
        for i in range(50):
            out = list(range(2500 + i * 50, 2500 + i * 50 + 49))
            writer.write(pa.record_batch([out], schema=schema))

    return tmpdir


# Make mock data for re-use (shared across all DCP tests)
_tmpdir = generate_sequential_multidata()
TEST_DATA_PATH = _tmpdir.name


def get_ckpt_path(subdir: str) -> str:
    """Get a checkpoint path under the test data directory."""
    ckpt_path = os.path.join(TEST_DATA_PATH, subdir)
    os.makedirs(ckpt_path, exist_ok=True)
    return ckpt_path


#### -------------------------    PIPELINE BUILDERS    ------------------------- ####


def pipeline(
    # Base reader
    path=None,
    rank=0,
    worldsize=1,
    delimiter=-1,
    seed=42,
    chunk=1000,
    logicals=10,
    # Sampling
    sample=True,
    datasets=["dataset_1", "dataset_2"],
    weights=[2, 1],
    # Packing
    pack=True,
    seqlen=100,
    npads=0,
    pad=-2,
    nbins=1,
    # Shuffling
    shuffle=True,
    window=10,
    # Tensor
    tensor=True,
):
    """Build a dataloader pipeline with configurable options."""
    if path is None:
        path = TEST_DATA_PATH

    data = ScalableReader(
        path,
        rank,
        worldsize,
        ArrowHandler(),
        delimiter,
        None,
        seed=seed,
        max_chunksize=chunk,
        n_logical_shards=logicals,
    )
    if sample:
        data = SamplingDataset(path, data, delimiter, datasets, weights)
    if pack:
        data = DocPackingDataset(data, seqlen, npads, delimiter, pad, nbins)
    if shuffle:
        data = ShuffleDataset(data, window)
    if tensor:
        data = PreprocessDataset(data, torch.tensor)
    return data


# Pre-configured pipeline shortcuts
basicdata = functools.partial(pipeline, sample=False, pack=False, shuffle=False)
basicdata1 = functools.partial(
    basicdata, path=os.path.join(TEST_DATA_PATH, "dataset_1")
)


#### -------------------------    DISTRIBUTED HELPERS    ------------------------- ####


def init_distributed(rank: int = 0, world_size: int = 1, port: str = "29500"):
    """
    Initialize a fake distributed environment for testing.

    Args:
        rank: The rank of this process
        world_size: Total number of processes
        port: Port for the master address (use different ports for different test files)

    Returns:
        DeviceMesh for the distributed environment
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
        )
    return dist.device_mesh.init_device_mesh("cpu", (world_size,))


def cleanup_distributed():
    """Cleanup the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
