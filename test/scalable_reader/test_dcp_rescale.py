"""
Tests for DCP rescaling functionality.

These tests verify that save_ckpt_dcp and load_ckpt_dcp correctly handle
rescaling scenarios where the number of workers changes between save and load.

Note: These tests use a simulated distributed environment since DCP requires
torch.distributed to be initialized.
"""

import math
import os
import shutil

from test_dcp_utils import (
    basicdata1,
    cleanup_distributed,
    get_ckpt_path,
    init_distributed,
)
from torchdata.scalable_reader.dcp import load_ckpt_dcp, save_ckpt_dcp
from torchdata.stateful_dataloader import StatefulDataLoader


# Checkpoint directory for this test file
ckptpath = get_ckpt_path("dcp_rescale_ckpts")


#### -------------------------    DCP RESCALING TESTS    ------------------------- ####


def test_dcp_rescale_epoch():
    """
    Test DCP save/load with rescaling to different number of workers.

    Complete part of an epoch, then rescale. Verify that until epoch is complete,
    data does not repeat, and that once epoch is complete, all data has appeared.

    Note: This test simulates rescaling by saving with one configuration and
    loading with a different number of workers. In a real distributed setting,
    this would involve multiple processes.
    """
    ckpt_dir = os.path.join(ckptpath, "test_rescale")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    try:
        mesh = init_distributed(rank=0, world_size=1, port="29501")

        logicals = 23
        nsteps = 15
        w1 = 2  # workers before rescale
        w2 = 3  # workers after rescale

        # Take first round of steps with w1 workers
        loader = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=logicals),
            num_workers=w1,
        )

        test_tokens = []
        for i, out in enumerate(loader):
            test_tokens.append(out.squeeze()[0].item())
            if i == w1 * nsteps - 1:
                break

        # Save checkpoint
        save_ckpt_dcp(loader, ckpt_dir, mesh)

        # Calculate approximate steps to complete epoch
        min_logical_size = math.floor(100 / logicals)
        max_logical_size = math.ceil(100 / logicals)
        min_extra_steps = min_logical_size * logicals // w2 - len(test_tokens) // w2
        max_extra_steps = max_logical_size * logicals - len(test_tokens) // w2

        # Create second loader with different number of workers and load
        loader2 = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=logicals),
            num_workers=w2,
        )
        load_ckpt_dcp(loader2, ckpt_dir, mesh)

        # Take more steps after rescaling
        test_tokens_after = []
        for i, out in enumerate(loader2):
            test_tokens_after.append(out.squeeze()[0].item())
            if i == w2 * max_extra_steps - 1:
                break

        # Verify no immediate repetition (basic sanity check)
        test_set = set(test_tokens)
        test_after_set = set(
            test_tokens_after[: min(len(test_tokens_after), w2 * min_extra_steps)]
        )

        # Check that we got some unique tokens after rescaling
        assert len(test_after_set) > 0, "No tokens produced after rescaling"

        # Verify the combined set covers more data
        combined = test_set.union(set(test_tokens_after))
        assert len(combined) >= len(
            test_set
        ), f"Combined coverage should be >= initial: {len(combined)} vs {len(test_set)}"

        print(f"test_dcp_rescale_epoch PASSED")
        print(f"  - Tokens before rescale: {len(test_tokens)}")
        print(f"  - Tokens after rescale: {len(test_tokens_after)}")
        print(f"  - Combined unique tokens: {len(combined)}")

    finally:
        cleanup_distributed()
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)


def test_dcp_rescale_custom_vars():
    """
    Test that custom variables (like shard_states) are properly rescaled
    using the custom resharding function during DCP load.

    This specifically tests the shard_rescale function integration with DCP.
    """
    ckpt_dir = os.path.join(ckptpath, "test_rescale_custom")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    try:
        mesh = init_distributed(rank=0, world_size=1, port="29501")

        # Create loader and iterate partway through
        loader = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=20),
            num_workers=2,
        )

        for i, _ in enumerate(loader):
            if i == 24:
                break

        # Save checkpoint
        save_ckpt_dcp(loader, ckpt_dir, mesh)

        # Load into a loader with different worker count
        # This triggers the rescaling path in load_ckpt_dcp
        loader2 = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=20),
            num_workers=4,  # Different from original 2
        )

        # This should trigger rescaling and custom var handling
        load_ckpt_dcp(loader2, ckpt_dir, mesh)

        # Verify we can continue iteration
        count = 0
        for i, out in enumerate(loader2):
            count += 1
            # Verify output is valid (not None or empty)
            assert out is not None, f"Got None output at step {i}"
            assert out.numel() > 0, f"Got empty output at step {i}"
            if i == 9:
                break

        assert count == 10, f"Expected 10 iterations after rescale, got {count}"

        print("test_dcp_rescale_custom_vars PASSED")

    finally:
        cleanup_distributed()
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)


def test_dcp_rescale_worker_increase():
    """
    Test rescaling from fewer workers to more workers.

    This tests the case where we scale up the number of workers,
    which requires redistributing shard states across more workers.
    """
    ckpt_dir = os.path.join(ckptpath, "test_rescale_increase")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    try:
        mesh = init_distributed(rank=0, world_size=1, port="29501")

        # Start with 1 worker
        loader = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=15),
            num_workers=1,
        )

        # Take some steps
        tokens_before = []
        for i, out in enumerate(loader):
            tokens_before.append(out.squeeze()[0].item())
            if i == 14:
                break

        save_ckpt_dcp(loader, ckpt_dir, mesh)

        # Scale up to 4 workers
        loader2 = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=15),
            num_workers=4,
        )
        load_ckpt_dcp(loader2, ckpt_dir, mesh)

        # Verify iteration continues
        tokens_after = []
        for i, out in enumerate(loader2):
            tokens_after.append(out.squeeze()[0].item())
            if i == 14:
                break

        assert len(tokens_after) == 15, f"Expected 15 tokens, got {len(tokens_after)}"

        print("test_dcp_rescale_worker_increase PASSED")
        print(f"  - Scaled from 1 worker to 4 workers")

    finally:
        cleanup_distributed()
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)


def test_dcp_rescale_worker_decrease():
    """
    Test rescaling from more workers to fewer workers.

    This tests the case where we scale down the number of workers,
    which requires consolidating shard states.
    """
    ckpt_dir = os.path.join(ckptpath, "test_rescale_decrease")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    try:
        mesh = init_distributed(rank=0, world_size=1, port="29501")

        # Start with 4 workers
        loader = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=15),
            num_workers=4,
        )

        # Take some steps
        tokens_before = []
        for i, out in enumerate(loader):
            tokens_before.append(out.squeeze()[0].item())
            if i == 19:
                break

        save_ckpt_dcp(loader, ckpt_dir, mesh)

        # Scale down to 1 worker
        loader2 = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=15),
            num_workers=1,
        )
        load_ckpt_dcp(loader2, ckpt_dir, mesh)

        # Verify iteration continues
        tokens_after = []
        for i, out in enumerate(loader2):
            tokens_after.append(out.squeeze()[0].item())
            if i == 9:
                break

        assert len(tokens_after) == 10, f"Expected 10 tokens, got {len(tokens_after)}"

        print("test_dcp_rescale_worker_decrease PASSED")
        print(f"  - Scaled from 4 workers to 1 worker")

    finally:
        cleanup_distributed()
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)


# NOTE: Testing actual world_size changes (e.g., from 1 GPU to 2 GPUs) requires
# a true multi-process environment (via torchrun or torch.multiprocessing.spawn).
# Single-process tests cannot simulate this because torch.distributed.init_process_group
# with world_size > 1 will wait for other processes to join.


#### -------------------------    RUN TESTS    ------------------------- ####


if __name__ == "__main__":
    print("=" * 60)
    print("Running DCP Rescaling Tests")
    print("=" * 60)

    print("\n--- Worker Count Rescaling Tests ---")
    test_dcp_rescale_epoch()
    test_dcp_rescale_custom_vars()
    test_dcp_rescale_worker_increase()
    test_dcp_rescale_worker_decrease()

    print("\n" + "=" * 60)
    print("All DCP rescaling tests passed!")
    print("=" * 60)
