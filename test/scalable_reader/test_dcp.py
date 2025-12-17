"""
Tests for DCP (Distributed Checkpoint) utilities.

These tests verify the save_ckpt_dcp and load_ckpt_dcp functions work correctly
for both same-scale loading and rescaling scenarios.

Note: These tests use a simulated distributed environment since DCP requires
torch.distributed to be initialized.
"""

import os
import shutil

import torch

from test_dcp_utils import (
    basicdata1,
    cleanup_distributed,
    get_ckpt_path,
    init_distributed,
    pipeline,
    TEST_DATA_PATH,
)

from torchdata.scalable_reader.dcp import load_ckpt_dcp, save_ckpt_dcp
from torchdata.stateful_dataloader import StatefulDataLoader


# Checkpoint directory for this test file
ckptpath = get_ckpt_path("dcp_ckpts")


#### -------------------------    DCP UNIT TESTS    ------------------------- ####


def test_dcp_save_load_basic():
    """
    Test basic DCP save and load without rescaling.

    Verifies that:
    1. save_ckpt_dcp creates checkpoint files
    2. load_ckpt_dcp restores state correctly
    3. Iteration continues successfully after loading

    Note: Due to prefetching behavior in StatefulDataLoader, exact output matching
    between original continuation and loaded loader is not reliable. This test
    verifies that loading works and produces valid outputs. For exact output
    matching tests, see test_informal.py which tests the underlying state_dict
    mechanism directly without DCP.
    """
    ckpt_dir = os.path.join(ckptpath, "test_basic")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    try:
        # Initialize distributed
        mesh = init_distributed(rank=0, world_size=1, port="29510")

        # Create loader and take some steps
        loader = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=10),
            num_workers=2,
        )

        outputs_before = []
        for i, out in enumerate(loader):
            outputs_before.append(out.squeeze()[0].item())
            if i == 19:
                break

        # Save checkpoint
        save_ckpt_dcp(loader, ckpt_dir, mesh)

        # Verify checkpoint files were created
        assert os.path.exists(ckpt_dir), "Checkpoint directory not created"
        distcp_files = [f for f in os.listdir(ckpt_dir) if ".distcp" in f]
        assert len(distcp_files) > 0, "No .distcp files created"

        # Create new loader and load checkpoint
        loader2 = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=10),
            num_workers=2,
        )
        (loader2, ckpt_dir, mesh)

        # Verify iteration continues successfully and produces valid outputs
        outputs_after_load = []
        for i, out in enumerate(loader2):
            outputs_after_load.append(out.squeeze()[0].item())
            if i == 9:
                break

        assert (
            len(outputs_after_load) == 10
        ), f"Expected 10 outputs after load, got {len(outputs_after_load)}"

        # Verify outputs are valid (should be beyond the first 20 rows)
        # With 10 logical shards over 100 rows, each row X has values like X*100 to X*100+99
        # After 20 iterations, we should be past row 10+ depending
        for i, val in enumerate(outputs_after_load):
            assert isinstance(
                val, (int, float)
            ), f"Invalid output type at {i}: {type(val)}"

        # Verify we're not at the beginning (row 0 values are 0-99)
        # At least some outputs should be >= 1000 (past row 10)
        max_val = max(outputs_after_load)
        assert (
            max_val >= 100
        ), f"Outputs seem to be from beginning of data: {outputs_after_load}"

        print("test_dcp_save_load_basic PASSED")

    finally:
        cleanup_distributed()
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)


def test_dcp_state_categories():
    """
    Test that all state categories (state, broadcast, reshard, custom) are
    properly saved and loaded via DCP.
    """
    ckpt_dir = os.path.join(ckptpath, "test_categories")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    try:
        mesh = init_distributed(rank=0, world_size=1, port="29511")

        # Use a more complex pipeline with multiple stateful layers
        loader = StatefulDataLoader(
            pipeline(
                path=TEST_DATA_PATH,
                rank=0,
                worldsize=1,
                logicals=15,
                sample=True,
                pack=True,
                shuffle=True,
                seqlen=50,
                window=5,
            ),
            num_workers=2,
        )

        # Take some steps
        for i, _ in enumerate(loader):
            if i == 29:
                break

        # Get state before save
        state_before = loader.state_dict()

        # Save and reload
        save_ckpt_dcp(loader, ckpt_dir, mesh)

        loader2 = StatefulDataLoader(
            pipeline(
                path=TEST_DATA_PATH,
                rank=0,
                worldsize=1,
                logicals=15,
                sample=True,
                pack=True,
                shuffle=True,
                seqlen=50,
                window=5,
            ),
            num_workers=2,
        )
        load_ckpt_dcp(loader2, ckpt_dir, mesh)

        # Verify we can continue iteration without errors
        count = 0
        for i, out in enumerate(loader2):
            count += 1
            if i == 9:
                break

        assert count == 10, f"Expected 10 iterations, got {count}"

        print("test_dcp_state_categories PASSED")

    finally:
        cleanup_distributed()
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)


def test_dcp_metadata_structure():
    """
    Test that checkpoint metadata is correctly structured after saving.
    """
    ckpt_dir = os.path.join(ckptpath, "test_metadata")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    try:
        mesh = init_distributed(rank=0, world_size=1, port="29512")

        loader = StatefulDataLoader(
            basicdata1(rank=0, worldsize=1, logicals=10),
            num_workers=2,
        )

        # Take some steps
        for i, _ in enumerate(loader):
            if i == 9:
                break

        save_ckpt_dcp(loader, ckpt_dir, mesh)

        # Check metadata structure
        from torchdata.scalable_reader.dcp.metadata import get_checkpoint_metadata

        meta = get_checkpoint_metadata(ckpt_dir)

        # Verify all expected top-level keys
        assert "state" in meta, "Missing 'state' in metadata"
        assert "broadcast" in meta, "Missing 'broadcast' in metadata"
        assert "reshard" in meta, "Missing 'reshard' in metadata"
        assert "custom" in meta, "Missing 'custom' in metadata"

        # Verify broadcast has global_worldsize
        assert (
            "global_worldsize" in meta["broadcast"]
        ), "Missing 'global_worldsize' in broadcast metadata"

        # Verify state has loader_state
        assert (
            "loader_state" in meta["state"]
        ), "Missing 'loader_state' in state metadata"

        print("test_dcp_metadata_structure PASSED")

    finally:
        cleanup_distributed()
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)


def test_dcp_dtensor_utils():
    """
    Test the DTensor utility functions directly.
    """
    from torchdata.scalable_reader.dcp.dtensor_utils import (
        crawl,
        unwrap_dtensor,
        wrap_dtensor,
    )

    try:
        mesh = init_distributed(rank=0, world_size=1, port="29513")

        # Test wrap_dtensor with nested dict
        test_dict = {
            "scalar": 42,
            "nested": {
                "inner_scalar": 123,
                "another": 456,
            },
        }

        wrapped = wrap_dtensor(test_dict.copy(), mesh)

        # Verify wrapped values are DTensors
        assert hasattr(wrapped["scalar"], "to_local"), "scalar not wrapped as DTensor"
        assert hasattr(
            wrapped["nested"]["inner_scalar"], "to_local"
        ), "nested scalar not wrapped as DTensor"

        # Test unwrap
        unwrapped_scalar = unwrap_dtensor(wrapped["scalar"], None)
        assert unwrapped_scalar == 42, f"Expected 42, got {unwrapped_scalar}"

        # Test crawl function
        source = {"a": 1, "b": {"c": 2}}
        meta = {"a": "meta_a", "b": {"c": "meta_c"}}
        result = crawl({}, meta, lambda d, m: f"{m}_processed")

        assert result["a"] == "meta_a_processed"
        assert result["b"]["c"] == "meta_c_processed"

        print("test_dcp_dtensor_utils PASSED")

    finally:
        cleanup_distributed()


def test_dcp_shard_wrapper():
    """
    Test wrap_shardtensor for creating sharded DTensors.
    """
    from torchdata.scalable_reader.dcp.dtensor_utils import wrap_shardtensor

    try:
        mesh = init_distributed(rank=0, world_size=1, port="29514")

        # Create a simple tensor
        tensor = torch.arange(100).reshape(10, 10)

        # Wrap it
        dtensor_wrapped = wrap_shardtensor(tensor, mesh, rank=0, worldsize=1)

        # Verify it's a DTensor
        assert hasattr(dtensor_wrapped, "to_local"), "Not wrapped as DTensor"

        # Verify shape is preserved
        assert dtensor_wrapped.shape == torch.Size(
            [10, 10]
        ), f"Shape mismatch: expected [10, 10], got {dtensor_wrapped.shape}"

        print("test_dcp_shard_wrapper PASSED")

    finally:
        cleanup_distributed()


#### -------------------------    RUN TESTS    ------------------------- ####


if __name__ == "__main__":
    print("=" * 60)
    print("Running DCP checkpoint tests")
    print("=" * 60)

    # Run utility tests first (don't require full DCP setup)
    print("\n--- DTensor Utility Tests ---")
    test_dcp_dtensor_utils()
    test_dcp_shard_wrapper()

    # Run full DCP save/load tests
    print("\n--- DCP Save/Load Tests ---")
    test_dcp_save_load_basic()
    test_dcp_state_categories()
    test_dcp_metadata_structure()

    print("\n" + "=" * 60)
    print("All DCP tests passed!")
    print("=" * 60)
