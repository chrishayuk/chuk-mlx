"""Tests for rollout buffer."""

import mlx.core as mx

from chuk_lazarus.data.rollout_buffer import (
    Episode,
    RolloutBuffer,
    Transition,
    compute_gae_inline,
)


class TestTransition:
    """Tests for Transition class."""

    def test_create_transition(self):
        """Test creating a transition."""
        transition = Transition(
            observation=[1, 2, 3],
            action=0,
            reward=1.0,
            done=False,
            log_prob=-0.5,
        )

        assert transition.observation == [1, 2, 3]
        assert transition.action == 0
        assert transition.reward == 1.0
        assert transition.done is False
        assert transition.log_prob == -0.5
        assert transition.value is None
        assert transition.hidden_state is None

    def test_create_transition_with_value(self):
        """Test creating transition with value."""
        transition = Transition(
            observation=[1, 2, 3],
            action=0,
            reward=1.0,
            done=False,
            log_prob=-0.5,
            value=0.8,
        )

        assert transition.value == 0.8

    def test_create_transition_with_hidden(self):
        """Test creating transition with hidden state."""
        hidden = mx.zeros((32,))
        transition = Transition(
            observation=[1, 2, 3],
            action=0,
            reward=1.0,
            done=False,
            log_prob=-0.5,
            hidden_state=hidden,
        )

        assert transition.hidden_state is hidden


class TestEpisode:
    """Tests for Episode class."""

    def test_create_empty_episode(self):
        """Test creating empty episode."""
        episode = Episode()

        assert episode.transitions == []
        assert episode.total_reward == 0.0
        assert episode.length == 0
        assert episode.info == {}

    def test_add_transition(self):
        """Test adding transitions to episode."""
        episode = Episode()

        t1 = Transition(
            observation=[1, 2, 3],
            action=0,
            reward=1.0,
            done=False,
            log_prob=-0.5,
        )
        t2 = Transition(
            observation=[4, 5, 6],
            action=1,
            reward=2.0,
            done=True,
            log_prob=-0.3,
        )

        episode.add(t1)
        assert episode.length == 1
        assert episode.total_reward == 1.0

        episode.add(t2)
        assert episode.length == 2
        assert episode.total_reward == 3.0

    def test_get_arrays(self):
        """Test converting episode to arrays."""
        episode = Episode()

        for i in range(3):
            episode.add(
                Transition(
                    observation=[i],
                    action=i,
                    reward=float(i),
                    done=i == 2,
                    log_prob=-0.1 * i,
                    value=0.5,
                )
            )

        arrays = episode.get_arrays()

        assert "rewards" in arrays
        assert "dones" in arrays
        assert "log_probs" in arrays
        assert "values" in arrays

        assert arrays["rewards"].tolist() == [0.0, 1.0, 2.0]
        assert arrays["dones"].tolist() == [0.0, 0.0, 1.0]


class TestComputeGAEInline:
    """Tests for compute_gae_inline function."""

    def test_basic_gae(self):
        """Test basic GAE computation."""
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]
        dones = [False, False, True]
        gamma = 0.99
        gae_lambda = 0.95

        advantages, returns = compute_gae_inline(rewards, values, dones, gamma, gae_lambda)

        assert advantages.shape == (3,)
        assert returns.shape == (3,)

    def test_gae_with_last_values(self):
        """Test GAE with bootstrap values."""
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]
        dones = [False, False, False]
        gamma = 0.99
        gae_lambda = 0.95
        last_values = mx.array([1.0])

        advantages, returns = compute_gae_inline(
            rewards, values, dones, gamma, gae_lambda, last_values
        )

        assert advantages.shape == (3,)
        assert returns.shape == (3,)

    def test_gae_single_step(self):
        """Test GAE with single step."""
        rewards = [1.0]
        values = [0.5]
        dones = [True]
        gamma = 0.99
        gae_lambda = 0.95

        advantages, returns = compute_gae_inline(rewards, values, dones, gamma, gae_lambda)

        assert advantages.shape == (1,)
        # With done=True, next_value=0, advantage = reward - value = 0.5
        assert abs(float(advantages[0]) - 0.5) < 0.01


class TestRolloutBuffer:
    """Tests for RolloutBuffer class."""

    def test_init(self):
        """Test buffer initialization."""
        buffer = RolloutBuffer(buffer_size=1024, gamma=0.99, gae_lambda=0.95)

        assert buffer.buffer_size == 1024
        assert buffer.gamma == 0.99
        assert buffer.gae_lambda == 0.95
        assert len(buffer) == 0

    def test_init_with_multiple_envs(self):
        """Test buffer initialization with multiple envs."""
        buffer = RolloutBuffer(num_envs=4)

        assert buffer.num_envs == 4
        assert len(buffer.current_episodes) == 4

    def test_add_transition(self):
        """Test adding single transition."""
        buffer = RolloutBuffer()

        buffer.add(
            observation=[1, 2, 3],
            action=0,
            reward=1.0,
            done=False,
            log_prob=-0.5,
            value=0.8,
        )

        assert len(buffer) == 1
        assert buffer.observations[0] == [1, 2, 3]
        assert buffer.actions[0] == 0
        assert buffer.rewards[0] == 1.0

    def test_add_transition_to_episode(self):
        """Test that transitions are tracked in episodes."""
        buffer = RolloutBuffer()

        # Add non-terminal transitions
        buffer.add([1], 0, 1.0, False, -0.5)
        buffer.add([2], 1, 2.0, False, -0.3)

        # Add terminal transition
        buffer.add([3], 0, 3.0, True, -0.2)

        # Should have completed one episode
        assert len(buffer.episodes) == 1
        assert buffer.episodes[0].total_reward == 6.0
        assert buffer.episodes[0].length == 3

    def test_reset(self):
        """Test buffer reset."""
        buffer = RolloutBuffer()

        # Add some data
        buffer.add([1], 0, 1.0, True, -0.5)

        # Reset
        buffer.reset()

        assert len(buffer) == 0
        assert buffer.observations == []
        assert buffer.episodes == []
        assert buffer.ptr == 0
        assert buffer.full is False

    def test_add_batch(self):
        """Test adding batch of transitions."""
        buffer = RolloutBuffer(num_envs=2)

        buffer.add_batch(
            observations=[[1], [2]],
            actions=[0, 1],
            rewards=[1.0, 2.0],
            dones=[False, False],
            log_probs=[-0.5, -0.3],
        )

        assert len(buffer) == 2

    def test_add_batch_with_values(self):
        """Test adding batch with values."""
        buffer = RolloutBuffer(num_envs=2)

        buffer.add_batch(
            observations=[[1], [2]],
            actions=[0, 1],
            rewards=[1.0, 2.0],
            dones=[False, False],
            log_probs=[-0.5, -0.3],
            values=[0.5, 0.6],
        )

        assert buffer.values == [0.5, 0.6]

    def test_compute_advantages(self):
        """Test computing advantages."""
        buffer = RolloutBuffer()

        # Add transitions
        for i in range(5):
            buffer.add([i], i, 1.0, i == 4, -0.5, value=0.5)

        buffer.compute_advantages()

        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.shape == (5,)
        assert buffer.returns.shape == (5,)

    def test_compute_advantages_with_bootstrap(self):
        """Test computing advantages with bootstrap values."""
        buffer = RolloutBuffer()

        # Add non-terminal transitions
        for i in range(5):
            buffer.add([i], i, 1.0, False, -0.5, value=0.5)

        last_values = mx.array([1.0])
        buffer.compute_advantages(last_values)

        assert buffer.advantages is not None
        assert buffer.returns is not None

    def test_get_batches(self):
        """Test generating batches."""
        buffer = RolloutBuffer()

        # Add transitions
        for i in range(10):
            buffer.add([i], i, 1.0, i == 9, -0.5, value=0.5)

        batches = list(buffer.get_batches(batch_size=3, shuffle=False))

        # 10 samples, batch_size=3 -> 4 batches
        assert len(batches) == 4

        # Check batch structure
        for batch in batches:
            assert "observations" in batch
            assert "actions" in batch
            assert "old_log_probs" in batch
            assert "advantages" in batch
            assert "returns" in batch

    def test_get_batches_auto_computes_advantages(self):
        """Test that get_batches computes advantages if needed."""
        buffer = RolloutBuffer()

        for i in range(5):
            buffer.add([i], i, 1.0, i == 4, -0.5, value=0.5)

        assert buffer.advantages is None

        list(buffer.get_batches(batch_size=2))

        assert buffer.advantages is not None

    def test_get_all(self):
        """Test getting all data."""
        buffer = RolloutBuffer()

        for i in range(5):
            buffer.add([i], i, float(i), i == 4, -0.1 * i, value=0.5)

        data = buffer.get_all()

        assert "observations" in data
        assert "actions" in data
        assert "old_log_probs" in data
        assert "advantages" in data
        assert "returns" in data
        assert "values" in data

    def test_get_episode_stats_empty(self):
        """Test episode stats with no episodes."""
        buffer = RolloutBuffer()

        stats = buffer.get_episode_stats()

        assert stats["num_episodes"] == 0
        assert stats["mean_reward"] == 0.0
        assert stats["mean_length"] == 0.0

    def test_get_episode_stats(self):
        """Test episode stats with episodes."""
        buffer = RolloutBuffer()

        # Episode 1: 3 transitions, total reward = 3
        buffer.add([1], 0, 1.0, False, -0.5)
        buffer.add([2], 0, 1.0, False, -0.5)
        buffer.add([3], 0, 1.0, True, -0.5)

        # Episode 2: 2 transitions, total reward = 5
        buffer.add([4], 0, 2.0, False, -0.5)
        buffer.add([5], 0, 3.0, True, -0.5)

        stats = buffer.get_episode_stats()

        assert stats["num_episodes"] == 2
        assert stats["mean_reward"] == 4.0  # (3 + 5) / 2
        assert stats["mean_length"] == 2.5  # (3 + 2) / 2
        assert stats["min_reward"] == 3.0
        assert stats["max_reward"] == 5.0

    def test_is_full(self):
        """Test is_full property."""
        buffer = RolloutBuffer(buffer_size=5)

        assert not buffer.is_full

        for i in range(5):
            buffer.add([i], i, 1.0, False, -0.5)

        assert buffer.is_full

    def test_len(self):
        """Test __len__ method."""
        buffer = RolloutBuffer()

        assert len(buffer) == 0

        buffer.add([1], 0, 1.0, False, -0.5)
        assert len(buffer) == 1

        buffer.add([2], 0, 1.0, False, -0.5)
        assert len(buffer) == 2
