"""Tests for gym episode stream."""

import pytest

from chuk_lazarus.data.batching.streaming import (
    EpisodeStatus,
    GymConfig,
    GymOutputMode,
    GymTransport,
    MockGymStream,
    SampleSource,
    StreamState,
)
from chuk_lazarus.data.batching.streaming.gym_stream import (
    GymEpisode,
    GymEpisodeStream,
    GymStep,
)


class MockTokenizer:
    """Simple mock tokenizer for testing."""

    @property
    def eos_token_id(self) -> int:
        return 2

    def encode(self, text: str) -> list[int]:
        # Simple word-based tokenization
        words = text.split()
        return [hash(word) % 1000 for word in words]


@pytest.fixture
def tokenizer():
    """Create a mock tokenizer."""
    return MockTokenizer()


class TestGymConfig:
    """Tests for GymConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = GymConfig()

        assert config.host == "localhost"
        assert config.port == 8023
        assert config.transport == GymTransport.TELNET
        assert config.output_mode == GymOutputMode.JSON
        assert config.connect_timeout == 10.0
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = GymConfig(
            host="puzzle.example.com",
            port=8080,
            transport=GymTransport.WEBSOCKET,
            puzzle_types=("arithmetic", "logic"),
            difficulty_range=(0.3, 0.7),
        )

        assert config.host == "puzzle.example.com"
        assert config.port == 8080
        assert config.transport == GymTransport.WEBSOCKET
        assert config.puzzle_types == ("arithmetic", "logic")
        assert config.difficulty_range == (0.3, 0.7)


class TestMockGymStream:
    """Tests for MockGymStream."""

    @pytest.mark.asyncio
    async def test_basic_iteration(self, tokenizer):
        """Test basic iteration over mock stream."""
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=5,
            steps_per_episode=3,
        )

        samples = []
        async for sample in stream:
            samples.append(sample)

        # 5 episodes * 3 steps = 15 samples
        assert len(samples) == 15

    @pytest.mark.asyncio
    async def test_sample_properties(self, tokenizer):
        """Test sample properties are correct."""
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=2,
            steps_per_episode=2,
        )

        samples = []
        async for sample in stream:
            samples.append(sample)

        for sample in samples:
            assert sample.source == SampleSource.GYM
            assert sample.is_gym_sample
            assert sample.episode_id is not None
            assert sample.step_index is not None
            assert sample.total_steps == 2
            assert sample.difficulty is not None

    @pytest.mark.asyncio
    async def test_success_rate(self, tokenizer):
        """Test success rate configuration."""
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=100,
            steps_per_episode=1,
            success_rate=0.7,
            seed=42,
        )

        success_count = 0
        async for sample in stream:
            if sample.success:
                success_count += 1

        # Should be roughly 70%
        assert 60 <= success_count <= 80

    @pytest.mark.asyncio
    async def test_difficulty_range(self, tokenizer):
        """Test difficulty range configuration."""
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=50,
            steps_per_episode=1,
            difficulty_range=(0.3, 0.7),
        )

        async for sample in stream:
            assert 0.3 <= sample.difficulty <= 0.7

    @pytest.mark.asyncio
    async def test_state_transitions(self, tokenizer):
        """Test stream state transitions."""
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=2,
            steps_per_episode=1,
        )

        assert stream.state == StreamState.IDLE

        async for _ in stream:
            assert stream.state == StreamState.STREAMING
            break

        # Exhaust stream
        async for _ in stream:
            pass

        assert stream.state == StreamState.EXHAUSTED

    @pytest.mark.asyncio
    async def test_metrics(self, tokenizer):
        """Test metrics accumulation."""
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=5,
            steps_per_episode=2,
        )

        async for _ in stream:
            pass

        metrics = stream.metrics
        assert metrics.samples_produced == 10
        assert metrics.episodes_completed == 5
        assert metrics.total_tokens > 0

    @pytest.mark.asyncio
    async def test_context_manager(self, tokenizer):
        """Test async context manager usage."""
        async with MockGymStream(
            tokenizer=tokenizer,
            num_episodes=3,
            steps_per_episode=1,
        ) as stream:
            samples = []
            async for sample in stream:
                samples.append(sample)

            assert len(samples) == 3

    @pytest.mark.asyncio
    async def test_reset(self, tokenizer):
        """Test stream reset."""
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=3,
            steps_per_episode=1,
        )

        # First pass
        count1 = 0
        async for _ in stream:
            count1 += 1

        await stream.reset()
        assert stream.state == StreamState.IDLE

        # Second pass
        count2 = 0
        async for _ in stream:
            count2 += 1

        assert count1 == count2

    @pytest.mark.asyncio
    async def test_episode_ids(self, tokenizer):
        """Test episode ID assignment."""
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=3,
            steps_per_episode=2,
        )

        episode_ids = set()
        async for sample in stream:
            episode_ids.add(sample.episode_id)

        # Should have 3 unique episode IDs
        assert len(episode_ids) == 3

    @pytest.mark.asyncio
    async def test_step_indices(self, tokenizer):
        """Test step index assignment."""
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=2,
            steps_per_episode=3,
        )

        samples = []
        async for sample in stream:
            samples.append(sample)

        # Check step indices for first episode
        first_ep_id = samples[0].episode_id
        first_ep_samples = [s for s in samples if s.episode_id == first_ep_id]

        assert len(first_ep_samples) == 3
        assert [s.step_index for s in first_ep_samples] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_reward_assignment(self, tokenizer):
        """Test reward assignment on final step."""
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=10,
            steps_per_episode=3,
            success_rate=1.0,  # All success
        )

        samples = []
        async for sample in stream:
            samples.append(sample)

        # Only final steps should have reward
        for sample in samples:
            if sample.step_index == 2:  # Final step
                assert sample.reward == 1.0
            else:
                assert sample.reward == 0.0


class TestGymStep:
    """Tests for GymStep model."""

    def test_basic_step(self):
        """Test basic step creation."""
        step = GymStep(
            step_index=0,
            observation="Solve this puzzle",
        )

        assert step.step_index == 0
        assert step.observation == "Solve this puzzle"
        assert step.action is None
        assert step.reward == 0.0
        assert step.done is False

    def test_step_with_action(self):
        """Test step with action."""
        step = GymStep(
            step_index=5,
            observation="Current state",
            action="place 1 5 7",
            reward=1.0,
            done=True,
            info={"valid": True},
        )

        assert step.step_index == 5
        assert step.action == "place 1 5 7"
        assert step.reward == 1.0
        assert step.done is True
        assert step.info == {"valid": True}


class TestGymEpisode:
    """Tests for GymEpisode model."""

    def test_basic_episode(self):
        """Test basic episode creation."""
        from datetime import datetime

        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 1, 30)

        steps = (
            GymStep(step_index=0, observation="Start"),
            GymStep(step_index=1, observation="Middle", action="move"),
            GymStep(step_index=2, observation="End", action="finish", done=True),
        )

        episode = GymEpisode(
            episode_id="ep_001",
            puzzle_type="sudoku",
            difficulty=0.5,
            steps=steps,
            total_reward=1.0,
            success=True,
            status=EpisodeStatus.SUCCESS,
            start_time=start,
            end_time=end,
        )

        assert episode.episode_id == "ep_001"
        assert episode.puzzle_type == "sudoku"
        assert episode.num_steps == 3
        assert episode.duration_seconds == 90.0
        assert episode.success is True

    def test_episode_properties(self):
        """Test episode computed properties."""
        from datetime import datetime

        steps = tuple(GymStep(step_index=i, observation=f"Step {i}") for i in range(5))

        episode = GymEpisode(
            episode_id="ep_002",
            puzzle_type="binary",
            difficulty=0.7,
            steps=steps,
            total_reward=0.0,
            success=False,
            status=EpisodeStatus.FAILURE,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 5, 0),
        )

        assert episode.num_steps == 5
        assert episode.duration_seconds == 300.0


class TestGymEpisodeStream:
    """Tests for GymEpisodeStream."""

    def test_stream_creation(self, tokenizer):
        """Test stream creation."""
        config = GymConfig()
        stream = GymEpisodeStream(
            config=config,
            tokenizer=tokenizer,
            dataset_id="test_gym",
            max_episodes=10,
        )

        assert stream.config.host == "localhost"
        assert stream.dataset_id == "test_gym"
        assert stream.max_episodes == 10
        assert stream.state == StreamState.IDLE

    def test_stream_properties(self, tokenizer):
        """Test stream property access."""
        config = GymConfig()
        stream = GymEpisodeStream(
            config=config,
            tokenizer=tokenizer,
        )

        assert stream.state == StreamState.IDLE
        assert stream.metrics is not None
        assert stream.metrics.samples_produced == 0

    def test_parse_response_json(self, tokenizer):
        """Test parsing JSON response."""
        config = GymConfig()
        stream = GymEpisodeStream(config=config, tokenizer=tokenizer)

        result = stream._parse_response('{"action": "place 1 1 5", "reward": 1.0}')
        assert result["action"] == "place 1 1 5"
        assert result["reward"] == 1.0

    def test_parse_response_invalid_json(self, tokenizer):
        """Test parsing invalid JSON falls back gracefully."""
        config = GymConfig()
        stream = GymEpisodeStream(config=config, tokenizer=tokenizer)

        result = stream._parse_response("Not valid JSON")
        assert result == {"raw": "Not valid JSON"}

    def test_episode_to_samples_empty_action(self, tokenizer):
        """Test episode conversion skips steps without action."""
        from datetime import datetime

        config = GymConfig()
        stream = GymEpisodeStream(config=config, tokenizer=tokenizer)

        steps = (
            GymStep(step_index=0, observation="Start"),  # No action
            GymStep(step_index=1, observation="Middle", action="move"),  # Has action
        )

        episode = GymEpisode(
            episode_id="ep_test",
            puzzle_type="test",
            difficulty=0.5,
            steps=steps,
            total_reward=1.0,
            success=True,
            status=EpisodeStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        samples = stream._episode_to_samples(episode)
        # Only step 1 should produce a sample (step 0 has no action)
        assert len(samples) == 1
        assert samples[0].step_index == 1

    def test_episode_to_samples_length_filter(self, tokenizer):
        """Test episode conversion filters by length."""
        from datetime import datetime

        config = GymConfig()
        stream = GymEpisodeStream(
            config=config,
            tokenizer=tokenizer,
            min_length=100,  # Very high minimum
        )

        steps = (GymStep(step_index=0, observation="Short", action="x"),)

        episode = GymEpisode(
            episode_id="ep_test",
            puzzle_type="test",
            difficulty=0.5,
            steps=steps,
            total_reward=1.0,
            success=True,
            status=EpisodeStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        samples = stream._episode_to_samples(episode)
        # Should be filtered out due to length
        assert len(samples) == 0


class TestGymConfigValidation:
    """Tests for GymConfig validation."""

    def test_output_modes(self):
        """Test all output modes."""
        for mode in GymOutputMode:
            config = GymConfig(output_mode=mode)
            assert config.output_mode == mode

    def test_transport_types(self):
        """Test all transport types."""
        for transport in GymTransport:
            config = GymConfig(transport=transport)
            assert config.transport == transport

    def test_puzzle_types(self):
        """Test puzzle type configuration."""
        config = GymConfig(
            puzzle_types=("sudoku", "binary", "kenken"),
        )
        assert config.puzzle_types == ("sudoku", "binary", "kenken")

    def test_timeout_validation(self):
        """Test timeout must be positive."""
        with pytest.raises(ValueError):
            GymConfig(connect_timeout=0)

        with pytest.raises(ValueError):
            GymConfig(read_timeout=-1)
