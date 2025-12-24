"""Tests for BatchPlan artifacts (Phase 4)."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from chuk_lazarus.data.batching import (
    BatchingConfig,
    BatchPlan,
    BatchPlanBuilder,
    BatchPlanMeta,
    EpochPlan,
    MicrobatchSpec,
    PadPolicy,
    load_batch_plan,
    save_batch_plan,
)


class TestMicrobatchSpec:
    """Tests for MicrobatchSpec."""

    def test_create_basic(self):
        mb = MicrobatchSpec(
            samples=["s1", "s2", "s3"],
            bucket_id=0,
            max_len=128,
            index=0,
        )
        assert mb.samples == ("s1", "s2", "s3")
        assert mb.bucket_id == 0
        assert mb.max_len == 128
        assert mb.batch_size == 3
        assert mb.packs is None

    def test_create_with_packs(self):
        mb = MicrobatchSpec(
            samples=["s1", "s2", "s3", "s4"],
            packs=[["s1", "s2"], ["s3", "s4"]],
            bucket_id=1,
            max_len=256,
            index=5,
        )
        assert mb.samples == ("s1", "s2", "s3", "s4")
        assert mb.packs == (("s1", "s2"), ("s3", "s4"))
        assert mb.batch_size == 4
        assert mb.num_packs == 2

    def test_list_to_tuple_conversion(self):
        mb = MicrobatchSpec(
            samples=["a", "b"],
            bucket_id=0,
            max_len=64,
            index=0,
        )
        assert isinstance(mb.samples, tuple)

    def test_immutable(self):
        mb = MicrobatchSpec(
            samples=["s1"],
            bucket_id=0,
            max_len=128,
            index=0,
        )
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            mb.bucket_id = 1


class TestEpochPlan:
    """Tests for EpochPlan."""

    def test_create(self):
        mbs = [
            MicrobatchSpec(samples=["s1", "s2"], bucket_id=0, max_len=128, index=0),
            MicrobatchSpec(samples=["s3", "s4"], bucket_id=0, max_len=128, index=1),
        ]
        epoch = EpochPlan(
            epoch=0,
            microbatches=mbs,
            seed=42,
            total_samples=4,
            total_tokens=400,
        )
        assert epoch.epoch == 0
        assert epoch.num_microbatches == 2
        assert epoch.total_samples == 4
        assert epoch.seed == 42

    def test_iteration(self):
        mbs = [MicrobatchSpec(samples=["s1"], bucket_id=0, max_len=128, index=i) for i in range(5)]
        epoch = EpochPlan(
            epoch=0,
            microbatches=mbs,
            seed=42,
            total_samples=5,
            total_tokens=500,
        )

        collected = list(epoch)
        assert len(collected) == 5
        assert all(isinstance(mb, MicrobatchSpec) for mb in collected)

    def test_getitem(self):
        mbs = [
            MicrobatchSpec(samples=[f"s{i}"], bucket_id=0, max_len=128, index=i) for i in range(3)
        ]
        epoch = EpochPlan(
            epoch=0,
            microbatches=mbs,
            seed=42,
            total_samples=3,
            total_tokens=300,
        )

        assert epoch[0].samples == ("s0",)
        assert epoch[1].samples == ("s1",)
        assert epoch[2].samples == ("s2",)


class TestBatchPlanMeta:
    """Tests for BatchPlanMeta."""

    def test_create_basic(self):
        meta = BatchPlanMeta(
            dataset_hash="abc123",
            tokenizer_hash="def456",
            bucket_edges=(128, 256, 512),
            overflow_max=1024,
            token_budget=4096,
            pad_policy=PadPolicy.PAD_TO_BUCKET,
            num_epochs=3,
            base_seed=42,
            created_at="2024-01-01T00:00:00",
        )
        assert meta.dataset_hash == "abc123"
        assert meta.bucket_edges == (128, 256, 512)
        assert meta.num_epochs == 3

    def test_create_from_config(self):
        config = BatchingConfig.predictable(
            token_budget=8192,
            bucket_edges=(256, 512),
            overflow_max=1024,
            seed=123,
        )
        meta = BatchPlanMeta.create(
            dataset_hash="data123",
            tokenizer_hash="tok456",
            batching_config=config,
            num_epochs=5,
        )

        assert meta.dataset_hash == "data123"
        assert meta.tokenizer_hash == "tok456"
        assert meta.token_budget == 8192
        assert meta.bucket_edges == (256, 512)
        assert meta.num_epochs == 5
        assert meta.base_seed == 123


class TestBatchPlan:
    """Tests for BatchPlan."""

    def _create_test_plan(self) -> BatchPlan:
        """Create a test plan with 2 epochs."""
        meta = BatchPlanMeta(
            dataset_hash="abc123",
            tokenizer_hash="def456",
            bucket_edges=(128, 256),
            overflow_max=512,
            token_budget=4096,
            pad_policy=PadPolicy.PAD_TO_BUCKET,
            num_epochs=2,
            base_seed=42,
            created_at="2024-01-01T00:00:00",
        )

        plan = BatchPlan(meta=meta)

        # Add epoch 0
        epoch0_mbs = [
            MicrobatchSpec(samples=["s1", "s2"], bucket_id=0, max_len=128, index=0),
            MicrobatchSpec(samples=["s3", "s4"], bucket_id=0, max_len=128, index=1),
            MicrobatchSpec(samples=["s5"], bucket_id=1, max_len=256, index=2),
        ]
        plan.add_epoch(
            EpochPlan(
                epoch=0,
                microbatches=tuple(epoch0_mbs),
                seed=42,
                total_samples=5,
                total_tokens=500,
            )
        )

        # Add epoch 1
        epoch1_mbs = [
            MicrobatchSpec(samples=["s3", "s1"], bucket_id=0, max_len=128, index=0),
            MicrobatchSpec(samples=["s2", "s4"], bucket_id=0, max_len=128, index=1),
            MicrobatchSpec(samples=["s5"], bucket_id=1, max_len=256, index=2),
        ]
        plan.add_epoch(
            EpochPlan(
                epoch=1,
                microbatches=tuple(epoch1_mbs),
                seed=43,
                total_samples=5,
                total_tokens=500,
            )
        )

        return plan

    def test_create(self):
        plan = self._create_test_plan()
        assert plan.num_epochs == 2
        assert plan.total_microbatches == 6

    def test_get_epoch(self):
        plan = self._create_test_plan()
        epoch0 = plan.get_epoch(0)
        assert epoch0.epoch == 0
        assert epoch0.num_microbatches == 3

    def test_get_epoch_missing(self):
        plan = self._create_test_plan()
        with pytest.raises(KeyError):
            plan.get_epoch(99)

    def test_fingerprint(self):
        plan = self._create_test_plan()
        fp = plan.fingerprint
        assert len(fp) == 16
        # Fingerprint should be stable
        assert plan.fingerprint == fp

    def test_iter_epoch(self):
        plan = self._create_test_plan()
        mbs = list(plan.iter_epoch(0))
        assert len(mbs) == 3
        assert all(isinstance(mb, MicrobatchSpec) for mb in mbs)


class TestBatchPlanSharding:
    """Tests for distributed sharding."""

    def _create_plan_for_sharding(self) -> BatchPlan:
        """Create a plan with 10 microbatches."""
        meta = BatchPlanMeta(
            dataset_hash="abc",
            tokenizer_hash="def",
            bucket_edges=(128,),
            overflow_max=256,
            token_budget=4096,
            pad_policy=PadPolicy.PAD_TO_BUCKET,
            num_epochs=1,
            base_seed=42,
            created_at="2024-01-01T00:00:00",
        )

        mbs = [
            MicrobatchSpec(samples=[f"s{i}"], bucket_id=0, max_len=128, index=i) for i in range(10)
        ]

        plan = BatchPlan(meta=meta)
        plan.add_epoch(
            EpochPlan(
                epoch=0,
                microbatches=tuple(mbs),
                seed=42,
                total_samples=10,
                total_tokens=1000,
            )
        )
        return plan

    def test_shard_2_workers(self):
        plan = self._create_plan_for_sharding()

        shard0 = plan.shard(rank=0, world_size=2)
        shard1 = plan.shard(rank=1, world_size=2)

        # Each shard gets 5 microbatches
        assert shard0.get_epoch(0).num_microbatches == 5
        assert shard1.get_epoch(0).num_microbatches == 5

        # Verify disjoint samples
        samples0 = set()
        for mb in shard0.iter_epoch(0):
            samples0.update(mb.samples)

        samples1 = set()
        for mb in shard1.iter_epoch(0):
            samples1.update(mb.samples)

        assert samples0.isdisjoint(samples1)
        assert len(samples0) + len(samples1) == 10

    def test_shard_4_workers(self):
        plan = self._create_plan_for_sharding()

        shards = [plan.shard(rank=r, world_size=4) for r in range(4)]

        # Check distribution
        for shard in shards:
            # 10 / 4 = 2-3 per shard
            assert 2 <= shard.get_epoch(0).num_microbatches <= 3

    def test_shard_invalid_rank(self):
        plan = self._create_plan_for_sharding()

        with pytest.raises(ValueError):
            plan.shard(rank=-1, world_size=4)

        with pytest.raises(ValueError):
            plan.shard(rank=4, world_size=4)


class TestBatchPlanResume:
    """Tests for resume support."""

    def test_iter_from_beginning(self):
        meta = BatchPlanMeta(
            dataset_hash="abc",
            tokenizer_hash="def",
            bucket_edges=(128,),
            overflow_max=256,
            token_budget=4096,
            pad_policy=PadPolicy.PAD_TO_BUCKET,
            num_epochs=2,
            base_seed=42,
            created_at="2024-01-01T00:00:00",
        )

        plan = BatchPlan(meta=meta)
        for ep in range(2):
            mbs = [
                MicrobatchSpec(samples=[f"e{ep}s{i}"], bucket_id=0, max_len=128, index=i)
                for i in range(3)
            ]
            plan.add_epoch(
                EpochPlan(
                    epoch=ep,
                    microbatches=tuple(mbs),
                    seed=42 + ep,
                    total_samples=3,
                    total_tokens=300,
                )
            )

        # Iterate from beginning
        results = list(plan.iter_from(epoch=0, microbatch_idx=0))
        assert len(results) == 6  # 3 + 3

    def test_iter_from_middle(self):
        meta = BatchPlanMeta(
            dataset_hash="abc",
            tokenizer_hash="def",
            bucket_edges=(128,),
            overflow_max=256,
            token_budget=4096,
            pad_policy=PadPolicy.PAD_TO_BUCKET,
            num_epochs=2,
            base_seed=42,
            created_at="2024-01-01T00:00:00",
        )

        plan = BatchPlan(meta=meta)
        for ep in range(2):
            mbs = [
                MicrobatchSpec(samples=[f"e{ep}s{i}"], bucket_id=0, max_len=128, index=i)
                for i in range(5)
            ]
            plan.add_epoch(
                EpochPlan(
                    epoch=ep,
                    microbatches=tuple(mbs),
                    seed=42 + ep,
                    total_samples=5,
                    total_tokens=500,
                )
            )

        # Resume from epoch 0, microbatch 2
        results = list(plan.iter_from(epoch=0, microbatch_idx=2))
        assert len(results) == 3 + 5  # 3 remaining from epoch 0 + all 5 from epoch 1


class TestBatchPlanIO:
    """Tests for save/load."""

    def _create_test_plan(self) -> BatchPlan:
        meta = BatchPlanMeta(
            dataset_hash="abc123",
            tokenizer_hash="def456",
            bucket_edges=(128, 256),
            overflow_max=512,
            token_budget=4096,
            pad_policy=PadPolicy.PAD_TO_BUCKET,
            num_epochs=2,
            base_seed=42,
            created_at="2024-01-01T00:00:00",
        )

        plan = BatchPlan(meta=meta)

        for ep in range(2):
            mbs = [
                MicrobatchSpec(samples=[f"e{ep}s{i}"], bucket_id=0, max_len=128, index=i)
                for i in range(3)
            ]
            plan.add_epoch(
                EpochPlan(
                    epoch=ep,
                    microbatches=tuple(mbs),
                    seed=42 + ep,
                    total_samples=3,
                    total_tokens=300,
                )
            )

        return plan

    def test_save_load_roundtrip(self):
        plan = self._create_test_plan()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plan"
            save_batch_plan(plan, path)

            # Check files created
            assert (path / "meta.json").exists()
            assert (path / "epoch_0.jsonl").exists()
            assert (path / "epoch_1.jsonl").exists()

            # Load and verify
            loaded = load_batch_plan(path)

            assert loaded.meta.dataset_hash == plan.meta.dataset_hash
            assert loaded.meta.tokenizer_hash == plan.meta.tokenizer_hash
            assert loaded.num_epochs == plan.num_epochs
            assert loaded.total_microbatches == plan.total_microbatches

            # Check epoch contents
            for ep in range(2):
                orig_mbs = list(plan.iter_epoch(ep))
                loaded_mbs = list(loaded.iter_epoch(ep))
                assert len(orig_mbs) == len(loaded_mbs)
                for orig_mb, loaded_mb in zip(orig_mbs, loaded_mbs):
                    assert orig_mb.samples == loaded_mb.samples
                    assert orig_mb.bucket_id == loaded_mb.bucket_id

    @pytest.mark.asyncio
    async def test_save_load_async(self):
        from chuk_lazarus.data.batching import (
            load_batch_plan_async,
            save_batch_plan_async,
        )

        plan = self._create_test_plan()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plan_async"
            await save_batch_plan_async(plan, path)

            # Load and verify
            loaded = await load_batch_plan_async(path)

            assert loaded.meta.dataset_hash == plan.meta.dataset_hash
            assert loaded.num_epochs == plan.num_epochs


class TestBatchPlanBuilder:
    """Tests for BatchPlanBuilder."""

    @pytest.mark.asyncio
    async def test_build_basic(self):
        lengths = {f"s{i}": 50 + i * 10 for i in range(20)}
        config = BatchingConfig.predictable(
            token_budget=1024,
            bucket_edges=(64, 128, 256),
            overflow_max=512,
            seed=42,
        )

        builder = BatchPlanBuilder(
            lengths=lengths,
            batching_config=config,
            dataset_hash="test_data",
            tokenizer_hash="test_tok",
        )

        plan = await builder.build(num_epochs=2)

        assert plan.num_epochs == 2
        assert plan.meta.dataset_hash == "test_data"
        assert plan.meta.tokenizer_hash == "test_tok"
        assert plan.meta.token_budget == 1024

        # Should have some microbatches
        assert plan.total_microbatches > 0

        # Check all samples are covered per epoch
        for ep in range(2):
            epoch_samples = set()
            for mb in plan.iter_epoch(ep):
                epoch_samples.update(mb.samples)
            # All samples should appear (unless skipped)
            assert len(epoch_samples) <= len(lengths)

    @pytest.mark.asyncio
    async def test_build_deterministic(self):
        lengths = {f"s{i}": 50 + i * 5 for i in range(15)}
        config = BatchingConfig.predictable(
            token_budget=512,
            seed=123,
        )

        builder = BatchPlanBuilder(
            lengths=lengths,
            batching_config=config,
            dataset_hash="data",
            tokenizer_hash="tok",
        )

        plan1 = await builder.build(num_epochs=2)
        plan2 = await builder.build(num_epochs=2)

        # Plans should be identical
        assert plan1.fingerprint == plan2.fingerprint
        assert plan1.total_microbatches == plan2.total_microbatches

        # Check microbatch order is identical
        for ep in range(2):
            mbs1 = list(plan1.iter_epoch(ep))
            mbs2 = list(plan2.iter_epoch(ep))
            for mb1, mb2 in zip(mbs1, mbs2):
                assert mb1.samples == mb2.samples


class TestIntegrationBatchPlan:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow: build -> save -> load -> shard."""
        # Build plan
        lengths = {f"sample_{i}": 100 + i * 10 for i in range(30)}
        config = BatchingConfig.predictable(
            token_budget=2048,
            bucket_edges=(128, 256, 512),
            overflow_max=1024,
        )

        builder = BatchPlanBuilder(
            lengths=lengths,
            batching_config=config,
            dataset_hash="integration_test",
            tokenizer_hash="tok_hash",
        )

        plan = await builder.build(num_epochs=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            path = Path(tmpdir) / "plan"
            save_batch_plan(plan, path)

            # Load
            loaded = load_batch_plan(path)

            # Shard for 2 workers
            shard0 = loaded.shard(rank=0, world_size=2)
            shard1 = loaded.shard(rank=1, world_size=2)

            # Verify shards are complete and disjoint
            for ep in range(3):
                all_samples = set()
                for mb in loaded.iter_epoch(ep):
                    all_samples.update(mb.samples)

                shard0_samples = set()
                for mb in shard0.iter_epoch(ep):
                    shard0_samples.update(mb.samples)

                shard1_samples = set()
                for mb in shard1.iter_epoch(ep):
                    shard1_samples.update(mb.samples)

                # Shards should be disjoint
                assert shard0_samples.isdisjoint(shard1_samples)

                # Together they should cover all
                assert shard0_samples | shard1_samples == all_samples
