# Batching & Training Infrastructure Roadmap

A phased roadmap for implementing bucketing, packing, batch plans, and dynamic gym integration in Lazarus. Designed for offline reproducibility, distributed-readiness, and future RL/curriculum learning.

## Goals

- **Throughput**: Reduce padding waste, maximize trained tokens per second
- **Reproducibility**: Deterministic batching with replay fingerprints
- **Distributed-ready**: Precomputed batch plans that shard cleanly across ranks
- **Extensibility**: Plug into `chuk-puzzles-gym` for online/RL training

## Phase Overview

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 0 | Lock the Interfaces | ✅ Complete | Canonical sample schema + fingerprinting |
| 1 | Bucketing + Token-Budget Batching | ✅ Complete | Offline length-based batching |
| 2 | Predictability Mode | ✅ Complete | Batch/shape invariance for reproducibility |
| 3 | Packing | ✅ Complete | Sequence concatenation with segment masks |
| 4 | BatchPlan Artifacts | ✅ Complete | Precomputed schedules for distributed training |
| 5 | Distributed Execution | ✅ Complete | Multi-rank sharding without rewrite |
| 5.5 | Offline Batch Generation | ✅ Complete | NPZ batch file creation with bucketing |
| 5.6 | Unified Batch Pipeline | ✅ Complete | BatchWriter/BatchReader in batching module |
| 6 | Dynamic Gym Integration | ✅ Complete | Stream + rolling windows for RL |
| 7 | Curriculum + Sampling Policies | Planned | Difficulty-based sampling, novelty, replay |

---

## Phase 0 — Lock the Interfaces

**Goal:** Make everything plug-compatible before adding features.

### Canonical Sample Schema

All samples must conform to a standard schema:

```python
@dataclass
class Sample:
    """Canonical sample format for training."""
    input_ids: list[int]
    loss_mask: list[int]           # SFT: 1 for completion, 0 for prompt
    segment_ids: list[int] | None  # For packed sequences (Phase 3)
    meta: SampleMeta

@dataclass
class SampleMeta:
    """Sample metadata for tracking and curriculum."""
    sample_id: str
    dataset_id: str
    episode_id: str | None = None
    reward: float | None = None
    difficulty: float | None = None
    source: str | None = None
```

For DPO, the `PreferenceSample` extends with `chosen` and `rejected` fields.

### Fingerprinting

Content-addressable hashing for reproducibility:

```python
def tokenizer_fingerprint(tokenizer) -> str:
    """Hash vocab, merges, chat template, and normalization rules."""
    ...

def dataset_fingerprint(dataset_path: Path, tokenizer_hash: str) -> str:
    """Hash dataset content + tokenizer for cache invalidation."""
    ...
```

### Deliverables

- [x] `Sample` dataclass with validation (Pydantic validators)
- [x] `SampleMeta` dataclass for metadata
- [x] `tokenizer_fingerprint()` → `compute_fingerprint()` in `data/tokenizers/fingerprint.py`
- [x] `dataset_fingerprint()` → `compute_dataset_fingerprint()` in `data/samples/schema.py`
- [x] Unit tests for schema validation (`tests/data/samples/test_schema.py`)

---

## Phase 1 — Bucketing + Token-Budget Batching

**Goal:** Reduce padding waste with length-based bucketing. Deterministic, offline.

### Length Cache

Pre-compute and cache sequence lengths:

```python
# lengths.jsonl format
{"sample_id": "train_00001", "tokenizer_hash": "abc123", "length": 247}
{"sample_id": "train_00002", "tokenizer_hash": "abc123", "length": 512}
```

### BucketSpec

Configure bucket boundaries:

```python
@dataclass
class BucketSpec:
    """Length bucket configuration."""
    edges: list[int]       # e.g., [128, 256, 512, 1024]
    overflow_bucket: int   # Max length for overflow (e.g., 2048)

    def get_bucket_id(self, length: int) -> int:
        """Return bucket index for a sequence length."""
        ...
```

### TokenBudgetBatchSampler

Form batches by token budget rather than sample count:

```python
class TokenBudgetBatchSampler:
    """Sample batches to maximize token budget utilization."""

    def __init__(
        self,
        lengths: dict[str, int],
        bucket_spec: BucketSpec,
        token_budget: int,
        seed: int,
    ):
        ...

    def __iter__(self) -> Iterator[list[str]]:
        """Yield batches of sample IDs."""
        # Deterministic shuffle: seed = base_seed + epoch
        # Bucket-local batching by token budget
        ...
```

### Metrics

Track efficiency improvements:

- `padding_waste`: Fraction of padding tokens in batch
- `effective_tokens_per_sec`: `sum(loss_mask) / batch_time`
- `batch_shape_histogram`: Bucket usage distribution

### CLI

```bash
# Build length cache
lazarus data lengths build \
    --dataset ./data/train.jsonl \
    --tokenizer "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --output ./cache/lengths.jsonl

# Train with token-budget batching
lazarus train sft \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --data ./data/train.jsonl \
    --bucket-edges 128,256,512,1024 \
    --token-budget 4096
```

### Deliverables

- [x] `LengthCache` class with build/load/save (`data/batching/generation/length_cache.py`)
- [x] `BucketSpec` dataclass (`data/batching/core/buckets.py`)
- [x] `TokenBudgetBatchSampler` implementation (`data/batching/core/sampler.py`)
- [x] Padding waste metrics (`data/batching/core/metrics.py`)
- [x] CLI commands: `lazarus data lengths build`, `lazarus data lengths stats`
- [x] Integration with existing trainers: `--bucket-edges`, `--token-budget` flags

---

## Phase 2 — Predictability Mode

**Goal:** Make runs repeatable and numerically stable across batching decisions.

### Mode Configuration

```python
class PadPolicy(Enum):
    PAD_TO_BUCKET = "pad_to_bucket"        # Predictable: fixed shapes
    PAD_TO_MAX_IN_BATCH = "pad_to_max"     # Throughput: dynamic shapes

@dataclass
class BatchingConfig:
    mode: Literal["predictable", "throughput"] = "throughput"
    pad_policy: PadPolicy = PadPolicy.PAD_TO_MAX_IN_BATCH
    bucket_max_lengths: dict[int, int] | None = None  # Predictable mode
```

### Replay Fingerprints

Hash batch contents for verification:

```python
def compute_batch_fingerprint(
    batches: list[Batch],
    n_batches: int = 100,
) -> str:
    """Hash first N microbatches' input_ids and loss_mask."""
    ...
```

Store in training logs for reproducibility verification.

### CLI

```bash
# Verify batching matches expected fingerprint
lazarus batch verify \
    --dataset ./data/train.jsonl \
    --config ./config/batch.yaml \
    --expected-fingerprint abc123def456
```

### Deliverables

- [x] `PadPolicy` enum (`data/batching/planning/predictability.py`)
- [x] `BatchingConfig` dataclass with `predictable()` and `throughput()` factories
- [x] `--predictable` CLI flag for `lazarus data batchplan build`
- [x] `compute_batch_fingerprint()` function (`data/batching/planning/predictability.py`)
- [x] `verify_batch_fingerprint()` function for replay verification
- [x] `BatchFingerprint` model with save/load support
- [x] `lazarus data batchplan verify` command
- [x] Fingerprint storage: `BatchWriter` includes fingerprint in NPZ output

---

## Phase 3 — Packing (Sequence Concatenation)

**Goal:** Reduce padding waste by concatenating sequences. Maintain correct semantics.

### Design Principles

1. **Packing in collator, not dataset** — Pack at batch formation time
2. **Deterministic** — Fixed ordering with deterministic tie-breaks
3. **Segment-aware attention** — Block-diagonal masks prevent cross-sample leakage

### Packing Strategy

```python
class PackingMode(Enum):
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"

def pack_sequences(
    samples: list[Sample],
    max_length: int,
    mode: PackingMode = PackingMode.FIRST_FIT,
) -> list[PackedSequence]:
    """Pack samples into sequences up to max_length."""
    ...

@dataclass
class PackedSequence:
    input_ids: list[int]
    loss_mask: list[int]
    segment_ids: list[int]  # 0, 0, 0, 1, 1, 1, 2, 2, ...
    sample_ids: list[str]   # Original sample IDs
```

### Segment-Aware Attention

Block-diagonal mask prevents attention across segment boundaries:

```python
def create_segment_attention_mask(segment_ids: list[int]) -> mx.array:
    """Create block-diagonal attention mask from segment IDs."""
    # segment_ids: [0, 0, 0, 1, 1, 2, 2, 2]
    # Returns mask where attention is blocked across segment boundaries
    ...
```

### CLI

```bash
lazarus train sft \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --data ./data/train.jsonl \
    --pack \
    --pack-max-len 2048 \
    --pack-mode first_fit
```

### Metrics

- `packing_efficiency`: Tokens used / total capacity
- Cross-sample leakage tests (unit tests verify attention isolation)

### Deliverables

- [x] `PackingMode` enum (FIRST_FIT, BEST_FIT, GREEDY)
- [x] `pack_sequences()` function with all three algorithms
- [x] `PackedSequence` dataclass with provenance tracking
- [x] `SequenceToPack` input wrapper
- [x] `PackingConfig` with separator support
- [x] `create_segment_attention_mask()` function (NumPy + MLX backends)
- [x] `PackingMetrics` and `compute_packing_metrics()` for efficiency tracking
- [x] Unit tests for all packing algorithms and attention isolation
- [x] Training CLI flags: `--pack`, `--pack-max-len`, `--pack-mode`
- [ ] Integration with attention layers (requires model changes)

---

## Phase 4 — BatchPlan Artifacts

**Goal:** Precomputed schedules for perfect reproducibility and distributed-readiness.

### BatchPlan Format

```python
@dataclass
class BatchPlanMeta:
    """Metadata for a batch plan."""
    dataset_hash: str
    tokenizer_hash: str
    bucket_edges: list[int]
    token_budget: int
    pack_config: PackConfig | None
    pad_policy: PadPolicy
    created_at: str
    version: str = "1.0"

@dataclass
class MicrobatchSpec:
    """Specification for a single microbatch."""
    samples: list[str]              # Sample IDs
    packs: list[list[str]] | None   # Packed sample groups (if packing)
    bucket_id: int
    max_len: int
```

### File Structure

```
batchplans/
├── meta.json           # BatchPlanMeta
├── epoch_0/
│   ├── microbatches.jsonl
│   └── stats.json
├── epoch_1/
│   └── ...
```

### CLI

```bash
# Build batch plan
lazarus batchplan build \
    --dataset ./data/train.jsonl \
    --tokenizer "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --bucket-edges 128,256,512,1024 \
    --token-budget 4096 \
    --epochs 3 \
    --output ./batchplans/run_001

# Train from batch plan
lazarus train sft \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --batchplan ./batchplans/run_001
```

### Deliverables

- [x] `BatchPlanMeta` dataclass with configuration capture
- [x] `MicrobatchSpec` dataclass with packing support
- [x] `EpochPlan` dataclass for epoch-level organization
- [x] `BatchPlan` class with build/load/save
- [x] `BatchPlanBuilder` for constructing plans from sampler
- [x] `shard()` method for distributed training
- [x] `iter_from()` for checkpoint resume
- [x] Async I/O support (`save_batch_plan_async`, `load_batch_plan_async`)
- [x] Unit tests for all components including sharding and resume
- [x] `lazarus data batchplan build` command
- [x] `lazarus data batchplan info` command
- [x] `--batchplan` flag for training CLI

---

## Phase 5 — Distributed Execution

**Goal:** Add rank/world_size without changing batching logic.

### Sharding Strategy

Shard by microbatch index across ranks:

```python
def shard_plan(
    plan: BatchPlan,
    rank: int,
    world_size: int,
) -> Iterator[MicrobatchSpec]:
    """Yield microbatches for this rank."""
    for i, mb in enumerate(plan.microbatches):
        if i % world_size == rank:
            yield mb
```

### Bucket Interleaving

During plan build, interleave buckets to balance work across ranks:

```python
def interleave_buckets(microbatches: list[MicrobatchSpec]) -> list[MicrobatchSpec]:
    """Interleave microbatches from different buckets for balanced sharding."""
    ...
```

### Resume Support

Checkpoint includes position in batch plan:

```python
@dataclass
class TrainingCheckpoint:
    epoch: int
    microbatch_idx: int
    model_state: dict
    optimizer_state: dict
```

### CLI

```bash
# Multi-process training (single machine)
lazarus train sft \
    --batchplan ./batchplans/run_001 \
    --rank 0 \
    --world-size 4

# In another process
lazarus train sft \
    --batchplan ./batchplans/run_001 \
    --rank 1 \
    --world-size 4
```

### Deliverables

- [x] `shard_plan()` function → `BatchPlan.shard()` in `data/batching/batch_plan.py`
- [x] `interleave_buckets()` function → `interleave_microbatches()` in `distributed/sharding.py`
- [x] `--rank` and `--world-size` CLI flags → `lazarus data batchplan info --rank --world-size`
- [x] `lazarus data batchplan shard` command for pre-sharding plans
- [x] Resume from checkpoint with microbatch index → `CheckpointPosition` + `iter_from_checkpoint()`
- [x] Rank synchronization utilities → `distributed/` module with `DistributedConfig`

---

## Phase 5.5 — Offline Batch Generation

**Goal:** Generate pre-tokenized batch files (NPZ) for fast training data loading.

### Overview

The `batch_generation` module provides offline batch file creation, complementing the runtime `batching` module:

- **`batch_generation`**: Offline tokenization → bucketing → NPZ files
- **`batching`**: Runtime batch scheduling (token-budget, packing, plans)

### BatchBase Class

Base class for batch creation workflows:

```python
from chuk_lazarus.data.batch_generation import BatchBase

class MyBatchGenerator(BatchBase):
    def tokenize_line(self, line: str) -> tuple[list[int], list[int], list[int]] | None:
        """Tokenize a single line into (input_tokens, target_tokens, attention_mask)."""
        data = json.loads(line)
        tokens = self.tokenizer.encode(data["text"])
        return tokens, tokens[1:] + [self.tokenizer.eos_token_id], [1] * len(tokens)

# Usage
generator = MyBatchGenerator(
    tokenizer=tokenizer,
    output_directory="./batches",
    file_prefix="train",
    max_sequence_length=2048,
    batch_size=8,
    bucket_spec=BucketSpec(edges=(128, 256, 512, 1024), overflow_max=2048),
)
generator.tokenize_and_batch(["data/train.jsonl"])
```

### Built-in Generators

```python
from chuk_lazarus.data.batch_generation import (
    PretrainBatchGenerator,  # Causal LM pre-training
    FineTuneBatch,           # Generic fine-tuning
    LLaMAFineTuneBatch,      # LLaMA instruction format
)

# Pre-training: next-token prediction
pretrain = PretrainBatchGenerator(
    tokenizer=tokenizer,
    output_directory="./batches/pretrain",
    file_prefix="pretrain",
    max_sequence_length=2048,
    batch_size=8,
)
pretrain.tokenize_and_batch(["corpus.txt"])

# LLaMA instruction fine-tuning
finetune = LLaMAFineTuneBatch(
    tokenizer=tokenizer,
    output_directory="./batches/finetune",
    file_prefix="sft",
    max_sequence_length=2048,
    batch_size=4,
)
finetune.tokenize_and_batch(["instructions.jsonl"])
```

### Bucketing Functions

Low-level bucketing utilities using `BucketSpec` from the batching module:

```python
from chuk_lazarus.data.batch_generation import (
    add_to_bucket,
    get_batch_from_bucket,
    iter_batches,
    get_bucket_stats,
)
from chuk_lazarus.data.batching import BucketSpec

bucket_spec = BucketSpec(edges=(128, 256, 512), overflow_max=1024)
buckets = {}

# Add samples to buckets
bucket_id = add_to_bucket(
    buckets, input_tokens, target_tokens, attention_mask,
    bucket_spec, sample_id="sample_001"
)

# Iterate batches with optional interleaving
for bucket_id, batch_data in iter_batches(buckets, batch_size=8, bucket_spec=bucket_spec):
    # Process batch
    ...

# Get statistics
stats = get_bucket_stats(buckets, bucket_spec)
# {0: {"count": 100, "avg_length": 95.2, "efficiency": 0.74}, ...}
```

### Utilities

```python
from chuk_lazarus.data.batch_generation import (
    pad_sequences,           # Pad to uniform length
    tokenize_dataset,        # Generator for tokenizing files
    get_line_text,           # Extract text from JSONL/JSON/plain
    visualize_sequences,     # Debug visualization
    calculate_memory_per_batch,  # Estimate memory usage
    batch_tokenize_and_pad,  # Batch tokenization helper
)

# Pad sequences
padded = pad_sequences(sequences, pad_id=0, max_length=256)

# Tokenize dataset
for result in tokenize_dataset(["data.jsonl"], tokenize_fn):
    input_tokens, target_tokens, attention_mask = result
    ...

# Memory estimation
mem = calculate_memory_per_batch(batch_size=8, seq_len=512, dtype="float32")
print(format_memory_size(mem))  # "16.00 MB"
```

### Output Format

Batches are saved as NPZ files:

```
batches/
├── train_batch_0001.npz
├── train_batch_0002.npz
└── ...

# Each NPZ contains:
# - input_tensor: (batch_size, seq_len) int32
# - target_tensor: (batch_size, seq_len) int32
# - attention_mask_tensor: (batch_size, seq_len) int32
```

### Integration with Batching Module

The `batch_generation` module uses `BucketSpec` from `batching` for consistent bucketing:

```python
from chuk_lazarus.data.batching import BucketSpec
from chuk_lazarus.data.batch_generation import BatchBase

# Same BucketSpec works in both modules
bucket_spec = BucketSpec(edges=(128, 256, 512, 1024), overflow_max=2048)

# Offline batch generation
generator = PretrainBatchGenerator(
    tokenizer=tokenizer,
    bucket_spec=bucket_spec,
    ...
)

# Runtime batch planning (different module, same bucket boundaries)
from chuk_lazarus.data.batching import TokenBudgetBatchSampler
sampler = TokenBudgetBatchSampler(
    lengths=lengths,
    bucket_spec=bucket_spec,
    token_budget=4096,
)
```

### Deliverables

- [x] `BatchBase` abstract class for batch workflows
- [x] `PretrainBatchGenerator` for causal LM pre-training
- [x] `FineTuneBatch` for generic fine-tuning
- [x] `LLaMAFineTuneBatch` for LLaMA instruction format
- [x] `add_to_bucket()`, `iter_batches()`, `get_bucket_stats()` functions
- [x] `pad_sequences()` utility
- [x] `tokenize_dataset()` generator
- [x] `calculate_memory_per_batch()` estimation
- [x] `visualize_sequences()` for debugging
- [x] Integration with `BucketSpec` from batching module
- [x] Unit tests (`tests/data/batch_generation/`)
- [x] CLI: `lazarus data batch generate` command
- [x] Async batch generation: `BatchWriter.write_all()` is sync, samples are processed incrementally

---

## Phase 6 — Dynamic Gym Integration

**Goal:** Plug into `chuk-puzzles-gym` for online/RL training with streaming data.

### SampleStream Interface

```python
class SampleStream(Protocol):
    """Interface for sample sources."""

    def __iter__(self) -> Iterator[Sample]:
        ...

class OfflineDatasetStream(SampleStream):
    """Stream from static dataset files."""
    ...

class GymEpisodeStream(SampleStream):
    """Stream from gym environment / puzzle server."""

    def __init__(
        self,
        env_name: str,
        host: str = "localhost",
        port: int = 23,  # telnet
    ):
        ...
```

### ReplayBuffer

Append-only bounded buffer for online learning:

```python
class ReplayBuffer:
    """Bounded replay buffer for online learning."""

    def __init__(self, max_size: int = 100_000):
        ...

    def add(self, sample: Sample) -> None:
        ...

    def sample(self, n: int) -> list[Sample]:
        ...

    def snapshot(self) -> list[Sample]:
        """Return current buffer contents for plan building."""
        ...
```

### Rolling BatchPlanWindow

Build plans over rolling windows:

```python
class RollingBatchPlanWindow:
    """Build plans over rolling buffer snapshots."""

    def __init__(
        self,
        buffer: ReplayBuffer,
        window_microbatches: int = 1000,
    ):
        ...

    def next_window(self) -> BatchPlan:
        """Snapshot buffer and build plan for next K microbatches."""
        ...
```

### CLI

```bash
# Connect to gym and run training
lazarus gym run \
    --env "puzzles-gym" \
    --host localhost \
    --port 23 \
    --buffer-size 100000

# Online training with rolling windows
lazarus train sft \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --online \
    --window-mbs 1000 \
    --gym-env "puzzles-gym"
```

### Deliverables

- [x] `SampleStream` protocol (`data/batching/streaming/protocols.py`)
- [x] `AsyncSampleStream` protocol (`data/batching/streaming/protocols.py`)
- [x] `StreamSample` model with provenance (`data/batching/streaming/types.py`)
- [x] `StreamMetrics` for tracking (`data/batching/streaming/types.py`)
- [x] `OfflineDatasetStream` implementation (`data/batching/streaming/offline_stream.py`)
- [x] `AsyncOfflineDatasetStream` implementation (`data/batching/streaming/offline_stream.py`)
- [x] `TelnetGymClient` for puzzle arcade telnet protocol (`data/batching/streaming/telnet_client.py`)
- [x] `TelnetClientConfig`, `PuzzleGame`, `PuzzleDifficulty` (`data/batching/streaming/telnet_client.py`)
- [x] `PuzzleObservation`, `PuzzleResult`, `PuzzleEpisode` models (`data/batching/streaming/telnet_client.py`)
- [x] `GymEpisodeStream` for puzzle arcade (`data/batching/streaming/gym_stream.py`)
- [x] `MockGymStream` for testing (`data/batching/streaming/gym_stream.py`)
- [x] `GymConfig`, `GymTransport`, `GymOutputMode` enums (`data/batching/streaming/gym_stream.py`)
- [x] `ReplayBuffer` class with priority sampling (`data/batching/streaming/replay_buffer.py`)
- [x] `ReplayBufferConfig`, `BufferEvictionPolicy` (`data/batching/streaming/replay_buffer.py`)
- [x] `BufferSnapshot` for batch planning (`data/batching/streaming/replay_buffer.py`)
- [x] `RollingBatchPlanWindow` class (`data/batching/streaming/rolling_window.py`)
- [x] `WindowConfig`, `WindowState` models (`data/batching/streaming/rolling_window.py`)
- [x] Unit tests (`tests/data/batching/streaming/`)
- [x] CLI commands: `lazarus gym run`, `lazarus gym info`
- [x] `--online`, `--gym-host`, `--gym-port`, `--buffer-size` flags for trainers
- [x] Example: `examples/batching/09_puzzle_arcade_integration.py`

---

## Phase 7 — Curriculum + Sampling Policies

**Goal:** Control buffer sampling for efficient curriculum learning.

### Sampling Policies

```python
class SamplingPolicy(Protocol):
    """Policy for sampling from replay buffer."""

    def sample(
        self,
        buffer: ReplayBuffer,
        n: int,
    ) -> list[Sample]:
        ...

class DifficultyBucketPolicy(SamplingPolicy):
    """Sample based on difficulty buckets."""

    def __init__(self, bucket_weights: dict[str, float]):
        # {"easy": 0.2, "medium": 0.5, "hard": 0.3}
        ...

class FailureReplayPolicy(SamplingPolicy):
    """Prioritize episodes where model failed."""

    def __init__(self, failure_weight: float = 2.0):
        ...

class NoveltyPolicy(SamplingPolicy):
    """Avoid repeating recently seen samples."""

    def __init__(self, novelty_window: int = 1000):
        ...

class LengthQuotaPolicy(SamplingPolicy):
    """Prevent short sequences from dominating."""

    def __init__(self, length_quotas: dict[int, float]):
        # {128: 0.1, 256: 0.2, 512: 0.4, 1024: 0.3}
        ...
```

### Policy Configuration

```yaml
# policy.yaml
policies:
  - type: difficulty_bucket
    weights:
      easy: 0.2
      medium: 0.5
      hard: 0.3
  - type: failure_replay
    failure_weight: 2.0
  - type: novelty
    novelty_window: 1000
  - type: length_quota
    quotas:
      128: 0.1
      256: 0.2
      512: 0.4
      1024: 0.3
```

### Metrics

- `success_rate_by_difficulty`: Track progress per difficulty level
- `replay_hit_rate`: How often samples are replayed
- `difficulty_distribution`: Current sampling distribution

### Deliverables

- [ ] `SamplingPolicy` protocol
- [ ] `DifficultyBucketPolicy` implementation
- [ ] `FailureReplayPolicy` implementation
- [ ] `NoveltyPolicy` implementation
- [ ] `LengthQuotaPolicy` implementation
- [ ] `policy.yaml` configuration format
- [ ] Curriculum metrics logging

---

## End-to-End Example

A complete pipeline from raw dataset to training-ready batches. See [`examples/batching/05_e2e_pipeline.py`](../examples/batching/05_e2e_pipeline.py) for runnable code.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      UNIFIED BATCH PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. DATASET              2. TOKENIZE              3. LENGTH INDEX       │
│   ┌─────────┐            ┌─────────┐              ┌─────────┐           │
│   │ train.  │  ───────►  │tokenizer│  ───────►   │{id: len}│           │
│   │ jsonl   │            │.encode()│              │  dict   │           │
│   └─────────┘            └─────────┘              └────┬────┘           │
│                                                        │                 │
│                                                        ▼                 │
│                              ┌──────────────────────────────────────┐   │
│                              │         4. BATCH PLAN                 │   │
│                              │  ┌─────────────────────────────────┐  │   │
│                              │  │ BatchingConfig + BatchPlanBuilder│  │   │
│                              │  │ → BatchPlan (universal IR)       │  │   │
│                              │  └─────────────────────────────────┘  │   │
│                              └──────────────┬───────────────────────┘   │
│                                             │                            │
│              ┌──────────────────────────────┼────────────────────────┐  │
│              ▼                              ▼                        ▼  │
│   ┌─────────────────┐          ┌─────────────────┐         ┌──────────┐│
│   │ BatchWriter     │          │ plan.iter_epoch │         │ .shard() ││
│   │ → NPZ cache     │          │ → stream        │         │ → ranks  ││
│   └────────┬────────┘          └────────┬────────┘         └────┬─────┘│
│            │                            │                       │      │
│            ▼                            │                       │      │
│   ┌─────────────────┐                   │                       │      │
│   │ BatchReader     │                   │                       │      │
│   │ → load NPZ      │                   │                       │      │
│   └────────┬────────┘                   │                       │      │
│            │                            │                       │      │
│            └────────────────────────────┼───────────────────────┘      │
│                                         ▼                               │
│                              ┌─────────────────────┐                    │
│                              │   TRAINING LOOP     │                    │
│                              │   model(batch)      │                    │
│                              └─────────────────────┘                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: `BatchPlan` is the universal intermediate representation. All paths (streaming, caching, distributed) consume the same plan.

### Stage 1: Dataset

```python
# train.jsonl format
{"id": "sample_0001", "instruction": "Explain gravity.", "response": "Gravity is..."}
{"id": "sample_0002", "instruction": "What is Python?", "response": "Python is..."}
```

### Stage 2: Tokenize & Compute Lengths

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def tokenize_sample(sample: dict) -> tuple[list[int], list[int], int]:
    """Returns (input_ids, loss_mask, length)."""
    prompt = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n"
    full_text = prompt + sample["response"]

    prompt_tokens = tokenizer.encode(prompt)
    full_tokens = tokenizer.encode(full_text) + [tokenizer.eos_token_id]
    loss_mask = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))

    return full_tokens, loss_mask, len(full_tokens)

# Build length index
lengths = {}
samples = {}
for line in open("train.jsonl"):
    sample = json.loads(line)
    input_ids, loss_mask, length = tokenize_sample(sample)
    lengths[sample["id"]] = length
    samples[sample["id"]] = {"input_ids": input_ids, "loss_mask": loss_mask}
```

### Stage 3: Build Batch Plan

```python
from chuk_lazarus.data.batching import BatchingConfig, BatchPlanBuilder

config = BatchingConfig.predictable(
    token_budget=4096,
    bucket_edges=(128, 256, 512),
    overflow_max=1024,
    seed=42,
)

plan = await BatchPlanBuilder(
    lengths=lengths,
    batching_config=config,
).build(num_epochs=3)

print(f"Plan: {plan.total_microbatches} batches, fingerprint={plan.fingerprint}")
```

### Stage 4: Training (Three Equivalent Paths)

All three paths use the **same `BatchPlan`** — choose based on your use case:

#### Path A: Stream (default, memory-efficient)

```python
for epoch in range(plan.num_epochs):
    for mb in plan.iter_epoch(epoch):
        batch = collate([samples[sid] for sid in mb.samples], mb.max_len)
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

#### Path B: Cache to NPZ (for repeated runs)

```python
from chuk_lazarus.data.batching import BatchWriter, BatchReader

# Write once
writer = BatchWriter(plan, output_dir="./batches", samples=samples)
await writer.write_all()

# Read many times
reader = BatchReader("./batches")
for epoch in range(plan.num_epochs):
    for batch in reader.iter_epoch(epoch):
        loss = model(batch)
        ...
```

#### Path C: Distributed (sharded across ranks)

```python
from chuk_lazarus.distributed import DistributedConfig

dist = DistributedConfig.from_env()
my_plan = plan.shard(rank=dist.rank, world_size=dist.world_size)

for epoch in range(my_plan.num_epochs):
    for mb in my_plan.iter_epoch(epoch):
        # Each rank processes different batches
        ...
```

### Complete Example

```python
#!/usr/bin/env python3
"""Unified batching pipeline."""

import asyncio
from chuk_lazarus.data.batching import BatchingConfig, BatchPlanBuilder

async def main():
    # 1. Load and tokenize
    lengths, samples = load_and_tokenize("train.jsonl", tokenizer)

    # 2. Build plan (single source of truth)
    plan = await BatchPlanBuilder(
        lengths=lengths,
        batching_config=BatchingConfig.predictable(token_budget=4096),
    ).build(num_epochs=3)

    # 3. Train (streaming path)
    for epoch in range(plan.num_epochs):
        for mb in plan.iter_epoch(epoch):
            batch = collate([samples[sid] for sid in mb.samples], mb.max_len)
            train_step(model, optimizer, batch)

if __name__ == "__main__":
    asyncio.run(main())
```

### When to Use Each Path

| Scenario | Path |
|----------|------|
| Default / large dataset | Stream (A) |
| Many training runs on same data | Cache (B) |
| Multi-GPU / multi-node | Distributed (C) |
| Debugging reproducibility | Cache + fingerprint verification |

---

## Phase 5.6 — Unified Batch Pipeline (Planned)

**Goal:** Unify `batch_generation` and `batching` into a single coherent pipeline.

### Current Problem

The two modules have overlapping responsibilities with subtle divergence:

| Aspect | `batch_generation` | `batching` |
|--------|-------------------|-----------|
| Purpose | Offline NPZ file creation | Runtime batch scheduling |
| Batch sizing | Sample count based | Token budget based |
| Output | NPZ files with tensors | BatchSpec/MicrobatchSpec |
| Async | Sync only | Async-native |
| Type safety | Dicts | Pydantic models |
| Reproducibility | None | Fingerprints + plans |

Both use `BucketSpec` for bucketing, but implement batch formation separately.

### Unified Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       UNIFIED BATCH PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐   │
│   │ TokenizedData │ ──► │ LengthIndex  │ ──► │ UnifiedBatchSampler  │   │
│   │ (lazy/eager)  │     │ {id: length} │     │ (token budget based) │   │
│   └──────────────┘     └──────────────┘     └──────────┬───────────┘   │
│                                                         │               │
│                                              ┌──────────▼───────────┐   │
│                                              │     BatchPlan        │   │
│                                              │  (MicrobatchSpec[])  │   │
│                                              └──────────┬───────────┘   │
│                                                         │               │
│              ┌──────────────────────────────────────────┼───────────┐   │
│              │                                          │           │   │
│              ▼                                          ▼           ▼   │
│   ┌─────────────────┐                    ┌─────────────────┐  ┌────────┐│
│   │ BatchWriter     │                    │ BatchIterator   │  │Sharding││
│   │ → NPZ files     │                    │ → stream during │  │→ ranks ││
│   │ (offline cache) │                    │   training      │  │        ││
│   └─────────────────┘                    └─────────────────┘  └────────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Design: Single BatchPlan, Multiple Consumers

The key insight is that **BatchPlan should be the universal intermediate representation**:

```python
# 1. Build plan (same for both paths)
plan = await BatchPlanBuilder(
    lengths=lengths,
    config=BatchingConfig.predictable(token_budget=4096),
).build(num_epochs=3)

# 2a. Runtime path: iterate during training
for mb in plan.iter_epoch(epoch=0):
    batch = collate(samples[mb.samples], max_len=mb.max_len)
    train_step(batch)

# 2b. Offline path: write NPZ files
writer = BatchWriter(plan, output_dir="./batches")
await writer.write_all(samples)  # Creates batch_0000.npz, etc.

# 2c. Load offline batches
for batch in BatchReader("./batches").iter_epoch(epoch=0):
    train_step(batch)  # Already collated numpy arrays
```

### Unified Components

#### 1. `BatchPlan` as Universal IR

```python
class MicrobatchSpec(BaseModel):
    """Universal batch specification."""
    samples: tuple[str, ...]
    bucket_id: int
    max_len: int
    index: int

    # Optional: pre-computed for offline caching
    input_ids: tuple[tuple[int, ...], ...] | None = None
    loss_mask: tuple[tuple[int, ...], ...] | None = None
```

#### 2. `BatchWriter` (replaces batch_generation core)

```python
class BatchWriter:
    """Write BatchPlan to NPZ files."""

    def __init__(
        self,
        plan: BatchPlan,
        output_dir: Path,
        collate_fn: Callable | None = None,
    ):
        self.plan = plan
        self.output_dir = output_dir
        self.collate_fn = collate_fn or default_collate

    async def write_epoch(self, epoch: int, samples: dict[str, Sample]):
        """Write all batches for an epoch."""
        for mb in self.plan.iter_epoch(epoch):
            batch_samples = [samples[sid] for sid in mb.samples]
            arrays = self.collate_fn(batch_samples, mb.max_len)
            np.savez(
                self.output_dir / f"epoch_{epoch}_batch_{mb.index:04d}.npz",
                **arrays,
                sample_ids=mb.samples,
                bucket_id=mb.bucket_id,
            )
```

#### 3. `BatchReader` (new)

```python
class BatchReader:
    """Read pre-written NPZ batches."""

    def __init__(self, batch_dir: Path):
        self.batch_dir = Path(batch_dir)
        self.plan = load_batch_plan(batch_dir / "plan")

    def iter_epoch(self, epoch: int) -> Iterator[dict[str, np.ndarray]]:
        """Iterate batches for an epoch."""
        for mb in self.plan.iter_epoch(epoch):
            path = self.batch_dir / f"epoch_{epoch}_batch_{mb.index:04d}.npz"
            yield dict(np.load(path))
```

#### 4. Unified Config

```python
class BatchingConfig(BaseModel):
    """Single config for all batching operations."""

    # Bucketing
    bucket_edges: tuple[int, ...] = (128, 256, 512, 1024)
    overflow_max: int = 2048
    min_length: int = 1

    # Batch formation (token budget is primary)
    token_budget: int = 4096

    # Padding
    pad_policy: PadPolicy = PadPolicy.PAD_TO_BUCKET

    # Reproducibility
    seed: int = 42
    drop_last: bool = False
    interleave_buckets: bool = True

    # Distributed
    world_size: int = 1
    rank: int = 0
```

### Migration Path

#### Phase 1: Add BatchWriter/BatchReader (non-breaking)

```python
# New: chuk_lazarus/data/batching/io.py
class BatchWriter:
    """Write BatchPlan to NPZ files."""
    ...

class BatchReader:
    """Read NPZ batches back."""
    ...
```

#### Phase 2: Deprecate batch_generation internals

```python
# batch_generation/__init__.py
from chuk_lazarus.data.batching import (
    BatchWriter,      # New unified writer
    BatchReader,      # New unified reader
    BucketSpec,       # Already shared
    BatchingConfig,   # Unified config
)

# Keep BatchBase for backwards compat, but delegate internally
class BatchBase:
    def tokenize_and_batch(self, input_files):
        # Build lengths
        lengths = self._compute_lengths(input_files)

        # Use unified planning
        plan = await BatchPlanBuilder(lengths, self.config).build(num_epochs=1)

        # Use unified writer
        writer = BatchWriter(plan, self.output_directory)
        await writer.write_epoch(0, self._load_samples(input_files))
```

#### Phase 3: Simplify to single module

Eventually, `batch_generation` becomes a thin wrapper or is merged entirely:

```python
# Single import for everything
from chuk_lazarus.data.batching import (
    # Planning
    BatchPlan,
    BatchPlanBuilder,
    MicrobatchSpec,

    # Config
    BatchingConfig,
    BucketSpec,

    # I/O
    BatchWriter,
    BatchReader,

    # Execution
    TokenBudgetBatchSampler,
)
```

### Benefits

1. **Single source of truth**: `BatchPlan` is the only batch specification
2. **Consistent bucketing**: Token-budget batching everywhere
3. **Reproducibility for free**: All paths get fingerprinting
4. **Simpler mental model**: Plan → (Write | Stream | Shard)
5. **Better testing**: Test plan once, all paths covered

### Deliverables

- [x] `BatchWriter` class for NPZ output (`data/batching/generation/io.py`)
- [x] `BatchReader` class for NPZ input (`data/batching/generation/io.py`)
- [x] `CollatedBatch` model for batch representation
- [x] `default_collate()` function for sample collation
- [x] `pad_sequences()` utility function
- [x] Unit tests (`tests/data/batching/test_io.py`)
- [x] CLI: `lazarus data batch generate` command
- [x] Unified exports in `data/batching/__init__.py`

**Note:** The original `batch_generation` module was never implemented separately - the unified approach was built directly into the `batching` module from the start.

---

## Implementation Order

Recommended sequence for maximum impact with minimal risk:

```
Phase 1 (Bucketing)           ✅ Complete
    ↓
Phase 4 (BatchPlan) ──→ Phase 2 (Predictability)    ✅ Complete
    ↓                         ↓
Phase 3 (Packing) ←──────────┘                      ✅ Complete
    ↓
Phase 5 (Distributed)                               ✅ Complete
    ↓
Phase 5.5 + 5.6 (Unified Pipeline)                  ✅ Complete
    ↓
Phase 6 (Gym Streams)                               ✅ Complete
    ↓
Phase 7 (Curriculum)          ← NEXT
```

**Rationale:**

1. **Phase 1** gives immediate throughput gains
2. **Phase 4** locks in reproducibility early
3. **Phase 2** adds fingerprinting for CI/CD verification
4. **Phase 3** further reduces padding waste
5. **Phase 5** scales to multi-GPU (uses existing batch plans)
6. **Phase 5.5 + 5.6** unifies batch_generation and batching into single coherent pipeline
7. **Phase 6** enables online/RL training with streaming data
8. **Phase 7** enables curriculum learning (builds on gym integration)

---

## Related Modules

- [`data/tokenizers/`](../src/chuk_lazarus/data/tokenizers/README.md) — Tokenizer toolkit
- [`data/batch_generation/`](../src/chuk_lazarus/data/batch_generation/) — Offline batch file generation
- [`data/batching/`](../src/chuk_lazarus/data/batching/) — Runtime batch scheduling
- [`distributed/`](../src/chuk_lazarus/distributed/) — Distributed training utilities
- [`training/`](../src/chuk_lazarus/training/) — Trainer implementations

## References

- [Thinking Machines Paper](https://arxiv.org/) — Predictability principles
- [Efficient Training of Language Models](https://arxiv.org/) — Packing and bucketing techniques
- [chuk-puzzles-gym](https://pypi.org/project/chuk-puzzles-gym/) — Multi-game puzzle gym for LLM training (24 puzzle types)
