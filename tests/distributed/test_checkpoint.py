"""Tests for checkpoint position utilities."""

import tempfile
from pathlib import Path

from chuk_lazarus.data.batching import (
    BatchPlan,
    BatchPlanMeta,
    EpochPlan,
    MicrobatchSpec,
    PadPolicy,
)
from chuk_lazarus.distributed import (
    CheckpointPosition,
    load_checkpoint_position,
    save_checkpoint_position,
)
from chuk_lazarus.distributed.checkpoint import iter_from_checkpoint


def create_test_plan(num_epochs: int = 2, batches_per_epoch: int = 5) -> BatchPlan:
    """Create a test batch plan."""
    meta = BatchPlanMeta(
        dataset_hash="test123",
        tokenizer_hash="tok456",
        bucket_edges=(128, 256),
        overflow_max=512,
        token_budget=4096,
        pad_policy=PadPolicy.PAD_TO_MAX_IN_BATCH,
        num_epochs=num_epochs,
        base_seed=42,
        created_at="2024-01-01T00:00:00Z",
    )

    plan = BatchPlan(meta=meta)

    for epoch in range(num_epochs):
        microbatches = [
            MicrobatchSpec(
                samples=(f"e{epoch}_s{i}",),
                bucket_id=0,
                max_len=128,
                index=i,
            )
            for i in range(batches_per_epoch)
        ]

        epoch_plan = EpochPlan(
            epoch=epoch,
            microbatches=tuple(microbatches),
            seed=42 + epoch,
            total_samples=batches_per_epoch,
            total_tokens=batches_per_epoch * 100,
        )
        plan.add_epoch(epoch_plan)

    return plan


class TestCheckpointPosition:
    """Tests for CheckpointPosition."""

    def test_create_position(self):
        """Test creating a checkpoint position."""
        pos = CheckpointPosition(epoch=1, microbatch_idx=5, global_step=100)
        assert pos.epoch == 1
        assert pos.microbatch_idx == 5
        assert pos.global_step == 100

    def test_to_dict(self):
        """Test converting to dictionary."""
        pos = CheckpointPosition(epoch=1, microbatch_idx=5, global_step=100)
        d = pos.to_dict()

        assert d["epoch"] == 1
        assert d["microbatch_idx"] == 5
        assert d["global_step"] == 100

    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {"epoch": 2, "microbatch_idx": 10, "global_step": 200}
        pos = CheckpointPosition.from_dict(d)

        assert pos.epoch == 2
        assert pos.microbatch_idx == 10
        assert pos.global_step == 200

    def test_from_dict_missing_global_step(self):
        """Test from_dict with missing global_step (backward compat)."""
        d = {"epoch": 2, "microbatch_idx": 10}
        pos = CheckpointPosition.from_dict(d)

        assert pos.epoch == 2
        assert pos.microbatch_idx == 10
        assert pos.global_step == 0


class TestSaveLoadCheckpointPosition:
    """Tests for save/load checkpoint position."""

    def test_save_and_load(self):
        """Test saving and loading checkpoint position."""
        pos = CheckpointPosition(epoch=1, microbatch_idx=5, global_step=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint_pos.json"
            save_checkpoint_position(pos, path)

            loaded = load_checkpoint_position(path)
            assert loaded.epoch == pos.epoch
            assert loaded.microbatch_idx == pos.microbatch_idx
            assert loaded.global_step == pos.global_step

    def test_save_creates_parent_dirs(self):
        """Test that save creates parent directories."""
        pos = CheckpointPosition(epoch=0, microbatch_idx=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dirs" / "checkpoint.json"
            save_checkpoint_position(pos, path)

            assert path.exists()


class TestIterFromCheckpoint:
    """Tests for iter_from_checkpoint function."""

    def test_iter_from_beginning(self):
        """Test iterating from beginning (no position)."""
        plan = create_test_plan(num_epochs=2, batches_per_epoch=5)

        items = list(iter_from_checkpoint(plan, position=None))

        # Should get all 10 items (2 epochs * 5 batches)
        assert len(items) == 10

        # First should be epoch 0, batch 0
        epoch, idx, mb = items[0]
        assert epoch == 0
        assert idx == 0

    def test_iter_from_middle_of_epoch(self):
        """Test iterating from middle of an epoch."""
        plan = create_test_plan(num_epochs=2, batches_per_epoch=5)

        # Start from epoch 0, batch 3
        pos = CheckpointPosition(epoch=0, microbatch_idx=3)
        items = list(iter_from_checkpoint(plan, position=pos))

        # Should get 2 from epoch 0 + 5 from epoch 1 = 7
        assert len(items) == 7

        # First should be epoch 0, batch 3
        epoch, idx, mb = items[0]
        assert epoch == 0
        assert idx == 3

    def test_iter_from_second_epoch(self):
        """Test iterating from start of second epoch."""
        plan = create_test_plan(num_epochs=2, batches_per_epoch=5)

        # Start from epoch 1, batch 0
        pos = CheckpointPosition(epoch=1, microbatch_idx=0)
        items = list(iter_from_checkpoint(plan, position=pos))

        # Should get only 5 items from epoch 1
        assert len(items) == 5

        # First should be epoch 1, batch 0
        epoch, idx, mb = items[0]
        assert epoch == 1
        assert idx == 0

    def test_iter_from_near_end(self):
        """Test iterating from near the end."""
        plan = create_test_plan(num_epochs=2, batches_per_epoch=5)

        # Start from epoch 1, batch 4 (last batch)
        pos = CheckpointPosition(epoch=1, microbatch_idx=4)
        items = list(iter_from_checkpoint(plan, position=pos))

        # Should get only 1 item
        assert len(items) == 1
        epoch, idx, mb = items[0]
        assert epoch == 1
        assert idx == 4
