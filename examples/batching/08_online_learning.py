#!/usr/bin/env python3
"""
Example 8: Online Learning with Gym Streams

Demonstrates the complete online learning infrastructure:
- MockGymStream for testing (simulates puzzle arcade)
- ReplayBuffer for experience storage
- RollingBatchPlanWindow for streaming batch plans
- Curriculum-aware difficulty tracking

Run:
    python examples/batching/08_online_learning.py
"""

import asyncio

from chuk_lazarus.data.batching.streaming import (
    BufferEvictionPolicy,
    GymConfig,
    GymOutputMode,
    GymTransport,
    MockGymStream,
    ReplayBuffer,
    ReplayBufferConfig,
    RollingBatchPlanWindow,
    WindowConfig,
)


class MockTokenizer:
    """Simple mock tokenizer for demonstration."""

    @property
    def eos_token_id(self) -> int:
        return 2

    def encode(self, text: str) -> list[int]:
        # Simple word-based tokenization
        words = text.split()
        return [hash(word) % 1000 for word in words]


async def demo_mock_gym_stream():
    """Demonstrate MockGymStream for testing."""
    print("\n" + "=" * 60)
    print("1. Mock Gym Stream")
    print("=" * 60)

    tokenizer = MockTokenizer()

    # Create mock gym stream
    stream = MockGymStream(
        tokenizer=tokenizer,
        num_episodes=5,
        steps_per_episode=3,
        difficulty_range=(0.2, 0.8),
        success_rate=0.7,
        seed=42,
    )

    print("   Config:")
    print("     num_episodes: 5")
    print("     steps_per_episode: 3")
    print("     difficulty_range: (0.2, 0.8)")
    print("     success_rate: 0.7")

    # Collect samples
    samples = []
    episode_ids = set()

    async with stream:
        async for sample in stream:
            samples.append(sample)
            if sample.episode_id:
                episode_ids.add(sample.episode_id)

    print("\n   Results:")
    print(f"     Total samples: {len(samples)}")
    print(f"     Unique episodes: {len(episode_ids)}")

    # Show sample properties
    print("\n   First sample:")
    s = samples[0]
    print(f"     sample_id:    {s.sample_id}")
    print(f"     episode_id:   {s.episode_id}")
    print(f"     step_index:   {s.step_index}")
    print(f"     total_steps:  {s.total_steps}")
    print(f"     difficulty:   {s.difficulty:.2f}")
    print(f"     source:       {s.source}")
    print(f"     is_gym_sample: {s.is_gym_sample}")

    # Check metrics
    metrics = stream.metrics
    print("\n   Stream metrics:")
    print(f"     samples_produced:    {metrics.samples_produced}")
    print(f"     episodes_completed:  {metrics.episodes_completed}")
    print(f"     total_tokens:        {metrics.total_tokens}")


async def demo_replay_buffer():
    """Demonstrate ReplayBuffer for experience storage."""
    print("\n" + "=" * 60)
    print("2. Replay Buffer")
    print("=" * 60)

    tokenizer = MockTokenizer()

    # Create replay buffer with FIFO eviction
    config = ReplayBufferConfig(
        max_size=50,
        eviction_policy=BufferEvictionPolicy.FIFO,
        seed=42,
        track_difficulty=True,
        track_success=True,
    )
    buffer = ReplayBuffer(config)

    print("   Buffer config:")
    print(f"     max_size: {config.max_size}")
    print(f"     eviction_policy: {config.eviction_policy}")

    # Create mock stream
    stream = MockGymStream(
        tokenizer=tokenizer,
        num_episodes=20,
        steps_per_episode=3,
        success_rate=0.6,
        seed=42,
    )

    # Collect samples into buffer
    async with stream:
        async for sample in stream:
            buffer.add(sample)

    print("\n   After collecting 60 samples:")
    print(f"     buffer.size:        {buffer.size}")
    print(f"     buffer.is_full:     {buffer.is_full}")
    print(f"     buffer.total_added: {buffer.total_added}")
    print(f"     buffer.total_evicted: {buffer.total_evicted}")

    # Statistics
    print("\n   Buffer statistics:")
    print(f"     mean_difficulty: {buffer.mean_difficulty:.2f}")
    print(f"     mean_reward:     {buffer.mean_reward:.2f}")
    print(f"     success_rate:    {buffer.success_rate:.1%}")

    # Sample from buffer
    sampled = buffer.sample(n=10)
    print("\n   Sampled 10 items (priority-weighted):")
    print(f"     All are replays: {all(s.is_replay for s in sampled)}")
    source_types = {s.source for s in sampled}
    print(f"     Source types: {source_types}")

    # Uniform sampling
    uniform = buffer.sample_uniform(n=10)
    print(f"     Uniform sample: {len(uniform)} items")

    # Filtering
    successful = buffer.filter_by_success(True)
    failed = buffer.filter_by_success(False)
    print("\n   Filtered by success:")
    print(f"     Successful: {len(successful)}")
    print(f"     Failed:     {len(failed)}")

    # Filter by difficulty
    easy = buffer.filter_by_difficulty(min_difficulty=0.0, max_difficulty=0.3)
    hard = buffer.filter_by_difficulty(min_difficulty=0.7, max_difficulty=1.0)
    print("\n   Filtered by difficulty:")
    print(f"     Easy (<0.3):  {len(easy)}")
    print(f"     Hard (>0.7):  {len(hard)}")


async def demo_priority_eviction():
    """Demonstrate priority-based eviction."""
    print("\n" + "=" * 60)
    print("3. Priority-Based Eviction")
    print("=" * 60)

    tokenizer = MockTokenizer()

    # Create buffer with priority eviction (evicts lowest priority first)
    config = ReplayBufferConfig(
        max_size=20,
        eviction_policy=BufferEvictionPolicy.PRIORITY,
        priority_alpha=0.8,  # High priority influence
        seed=42,
    )
    buffer = ReplayBuffer(config)

    print("   Eviction policy: PRIORITY")
    print(f"   Priority alpha: {config.priority_alpha}")

    # Add samples
    stream = MockGymStream(
        tokenizer=tokenizer,
        num_episodes=10,
        steps_per_episode=3,
        seed=42,
    )

    async with stream:
        async for sample in stream:
            # Use difficulty as priority (harder = higher priority)
            priority = sample.difficulty if sample.difficulty else 0.5
            buffer.add(sample, priority=priority)

    print("\n   After adding 30 samples to size-20 buffer:")
    print(f"     buffer.size: {buffer.size}")
    print(f"     evicted:     {buffer.total_evicted}")

    # Check difficulty distribution (should favor higher difficulty)
    difficulties = [s.difficulty for s in buffer.get_all() if s.difficulty]
    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0

    print(f"   Average difficulty in buffer: {avg_difficulty:.2f}")
    print("   (Expected to be higher due to priority eviction)")


async def demo_rolling_window():
    """Demonstrate RollingBatchPlanWindow for streaming batch plans."""
    print("\n" + "=" * 60)
    print("4. Rolling Batch Plan Window")
    print("=" * 60)

    tokenizer = MockTokenizer()

    # Create buffer
    buffer_config = ReplayBufferConfig(max_size=1000, seed=42)
    buffer = ReplayBuffer(buffer_config)

    # Fill buffer with initial samples
    stream = MockGymStream(
        tokenizer=tokenizer,
        num_episodes=50,
        steps_per_episode=3,
        seed=42,
    )

    async with stream:
        async for sample in stream:
            buffer.add(sample)

    print(f"   Initial buffer: {buffer.size} samples")

    # Create rolling window
    window_config = WindowConfig(
        window_microbatches=10,
        min_samples=50,
        bucket_edges=(128, 256),
        overflow_max=512,
        token_budget=512,
    )

    window = RollingBatchPlanWindow(
        buffer=buffer,
        config=window_config,
    )

    print("\n   Window config:")
    print(f"     window_microbatches: {window_config.window_microbatches}")
    print(f"     min_samples:         {window_config.min_samples}")
    print(f"     token_budget:        {window_config.token_budget}")

    # Build first window
    await window.build_next_window()
    print("\n   First window built:")
    print(f"     has_window: {window.has_window}")
    print(f"     windows_built: {window.total_windows}")

    if window.has_window:
        state = window.current_state
        print(f"     microbatches: {state.plan_microbatches}")

    # Iterate through window
    mb_count = 0
    for mb in window.iter_current():
        mb_count += 1
        if mb_count <= 3:
            print(f"     Microbatch {mb_count}: {len(mb.samples)} samples")

    print(f"     Total microbatches processed: {mb_count}")

    # Check if rebuild needed
    print(f"\n   Should rebuild: {window.should_build_next}")

    # Build next window
    await window.build_next_window()
    print(f"   Windows built total: {window.total_windows}")


async def demo_gym_config():
    """Demonstrate GymConfig for real server connections."""
    print("\n" + "=" * 60)
    print("5. Gym Configuration")
    print("=" * 60)

    # Show available transports
    print("   Available transports:")
    for t in GymTransport:
        print(f"     - {t.value}")

    print("\n   Available output modes:")
    for m in GymOutputMode:
        print(f"     - {m.value}")

    # Create config for puzzle arcade
    config = GymConfig(
        host="localhost",
        port=8023,
        transport=GymTransport.TELNET,
        output_mode=GymOutputMode.JSON,
        connect_timeout=10.0,
        max_retries=3,
        puzzle_types=("arithmetic", "logic"),
        difficulty_range=(0.3, 0.7),
    )

    print("\n   Sample config for puzzle arcade:")
    print(f"     host:            {config.host}")
    print(f"     port:            {config.port}")
    print(f"     transport:       {config.transport}")
    print(f"     output_mode:     {config.output_mode}")
    print(f"     connect_timeout: {config.connect_timeout}s")
    print(f"     max_retries:     {config.max_retries}")
    print(f"     puzzle_types:    {config.puzzle_types}")
    print(f"     difficulty:      {config.difficulty_range}")


async def demo_online_training_loop():
    """Demonstrate a complete online training loop."""
    print("\n" + "=" * 60)
    print("6. Online Training Loop (Simulation)")
    print("=" * 60)

    tokenizer = MockTokenizer()

    # Initialize components
    buffer = ReplayBuffer(ReplayBufferConfig(max_size=500, seed=42))

    window = RollingBatchPlanWindow(
        buffer=buffer,
        config=WindowConfig(
            window_microbatches=5,
            min_samples=5,  # Low threshold for demo
            bucket_edges=(128,),
            overflow_max=256,
            token_budget=256,
        ),
    )

    # Simulate training loop
    print("   Simulating training loop:")
    training_steps = 0
    max_steps = 20

    # Create a long-running stream
    stream = MockGymStream(
        tokenizer=tokenizer,
        num_episodes=100,
        steps_per_episode=2,
        seed=42,
    )

    async with stream:
        sample_iter = stream.__aiter__()

        while training_steps < max_steps:
            # Collect new samples
            for _ in range(5):
                try:
                    sample = await sample_iter.__anext__()
                    buffer.add(sample)
                except StopAsyncIteration:
                    break

            # Build/rebuild batch plan if needed
            if not window.has_window or window.should_build_next:
                await window.build_next_window()

            # Process microbatches
            if window.has_window:
                for mb in window.iter_current():
                    training_steps += 1
                    if training_steps <= 5:
                        print(f"     Step {training_steps}: batch of {len(mb.samples)} samples")
                    if training_steps >= max_steps:
                        break

    print("\n   Training summary:")
    print(f"     Total steps:       {training_steps}")
    print(f"     Buffer size:       {buffer.size}")
    print(f"     Windows built:     {window.total_windows}")
    print(f"     Total microbatches: {window.total_microbatches}")
    print(f"     Mean difficulty:   {buffer.mean_difficulty:.2f}")
    print(f"     Success rate:      {buffer.success_rate:.1%}")


async def main():
    print("=" * 60)
    print("Online Learning Demo")
    print("=" * 60)

    # Mock gym stream
    await demo_mock_gym_stream()

    # Replay buffer
    await demo_replay_buffer()

    # Priority eviction
    await demo_priority_eviction()

    # Rolling window
    await demo_rolling_window()

    # Gym config
    await demo_gym_config()

    # Online training loop
    await demo_online_training_loop()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
