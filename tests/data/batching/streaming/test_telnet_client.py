"""Tests for telnet client models and parsing logic."""

import pytest

from chuk_lazarus.data.batching.streaming import (
    DifficultyProfile,
    PuzzleDifficulty,
    PuzzleGame,
    PuzzleObservation,
    PuzzleResult,
    TelnetClientConfig,
)
from chuk_lazarus.data.batching.streaming.telnet_client import (
    PuzzleCompletion,
    PuzzleEpisode,
)


class TestTelnetClientConfig:
    """Tests for TelnetClientConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = TelnetClientConfig()

        assert config.host == "localhost"
        assert config.port == 8023
        assert config.connect_timeout == 10.0
        assert config.read_timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = TelnetClientConfig(
            host="puzzle.example.com",
            port=8080,
            connect_timeout=5.0,
            read_timeout=15.0,
            max_retries=5,
            retry_delay=2.0,
        )

        assert config.host == "puzzle.example.com"
        assert config.port == 8080
        assert config.connect_timeout == 5.0
        assert config.read_timeout == 15.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0

    def test_frozen(self):
        """Test config is frozen."""
        from pydantic import ValidationError

        config = TelnetClientConfig()
        with pytest.raises(ValidationError):
            config.host = "other"


class TestPuzzleEnums:
    """Tests for puzzle enums."""

    def test_puzzle_game_values(self):
        """Test PuzzleGame enum values."""
        assert PuzzleGame.SUDOKU.value == "sudoku"
        assert PuzzleGame.KENKEN.value == "kenken"
        assert PuzzleGame.BINARY.value == "binary"
        assert PuzzleGame.FUTOSHIKI.value == "futoshiki"
        assert PuzzleGame.MASTERMIND.value == "mastermind"
        assert PuzzleGame.MINESWEEPER.value == "minesweeper"

    def test_puzzle_game_count(self):
        """Test we have expected number of puzzle types."""
        # Should have 22 puzzle types
        assert len(PuzzleGame) == 22

    def test_puzzle_difficulty_values(self):
        """Test PuzzleDifficulty enum values."""
        assert PuzzleDifficulty.EASY.value == "easy"
        assert PuzzleDifficulty.MEDIUM.value == "medium"
        assert PuzzleDifficulty.HARD.value == "hard"


class TestDifficultyProfile:
    """Tests for DifficultyProfile."""

    def test_defaults(self):
        """Test default profile."""
        profile = DifficultyProfile()

        assert profile.logic_depth == 1
        assert profile.branching_factor == 1.0
        assert profile.state_observability == 1.0
        assert profile.constraint_density == 0.5

    def test_custom_profile(self):
        """Test custom profile."""
        profile = DifficultyProfile(
            logic_depth=5,
            branching_factor=3.5,
            state_observability=0.7,
            constraint_density=0.8,
        )

        assert profile.logic_depth == 5
        assert profile.branching_factor == 3.5
        assert profile.state_observability == 0.7
        assert profile.constraint_density == 0.8

    def test_from_dict(self):
        """Test creating from dict (as from JSON)."""
        data = {
            "logic_depth": 7,
            "branching_factor": 2.0,
            "state_observability": 0.9,
            "constraint_density": 0.6,
        }
        profile = DifficultyProfile(**data)

        assert profile.logic_depth == 7
        assert profile.branching_factor == 2.0

    def test_extra_fields_ignored(self):
        """Test extra fields are ignored."""
        data = {
            "logic_depth": 3,
            "unknown_field": "value",
        }
        profile = DifficultyProfile(**data)
        assert profile.logic_depth == 3


class TestPuzzleObservation:
    """Tests for PuzzleObservation."""

    def test_minimal_observation(self):
        """Test minimal observation."""
        obs = PuzzleObservation(
            type="observation",
            game="sudoku",
            difficulty="easy",
        )

        assert obs.type == "observation"
        assert obs.game == "sudoku"
        assert obs.difficulty == "easy"
        assert obs.seed == 0
        assert obs.moves == 0
        assert obs.is_complete is False

    def test_full_observation(self):
        """Test full observation with all fields."""
        grid = [[0, 1, 2], [3, 0, 4], [5, 6, 0]]
        profile = DifficultyProfile(logic_depth=3)

        obs = PuzzleObservation(
            type="observation",
            game="sudoku",
            difficulty="medium",
            seed=12345,
            moves=5,
            invalid_moves=1,
            hints_used=2,
            hints_remaining=3,
            optimal_steps=15,
            is_complete=False,
            grid=grid,
            grid_display="1 2 3\n4 5 6\n7 8 9",
            difficulty_profile=profile,
        )

        assert obs.seed == 12345
        assert obs.moves == 5
        assert obs.invalid_moves == 1
        assert obs.hints_used == 2
        assert obs.hints_remaining == 3
        assert obs.optimal_steps == 15
        assert obs.grid == grid
        assert obs.difficulty_profile.logic_depth == 3

    def test_from_json_response(self):
        """Test creating from JSON response."""
        data = {
            "type": "observation",
            "game": "binary",
            "difficulty": "hard",
            "seed": 42,
            "moves": 10,
            "invalid_moves": 2,
            "hints_used": 1,
            "hints_remaining": 4,
            "is_complete": False,
            "grid": [[0, 1], [1, 0]],
        }
        obs = PuzzleObservation(**data)

        assert obs.game == "binary"
        assert obs.difficulty == "hard"
        assert obs.grid == [[0, 1], [1, 0]]


class TestPuzzleResult:
    """Tests for PuzzleResult."""

    def test_success_result(self):
        """Test successful result."""
        result = PuzzleResult(
            type="result",
            success=True,
            code="OK",
            message="Move accepted",
        )

        assert result.success is True
        assert result.code == "OK"
        assert result.message == "Move accepted"

    def test_failure_result(self):
        """Test failure result."""
        result = PuzzleResult(
            type="result",
            success=False,
            code="INVALID",
            message="Invalid move: cell already filled",
        )

        assert result.success is False
        assert result.code == "INVALID"

    def test_with_state(self):
        """Test result with updated state."""
        obs = PuzzleObservation(
            type="observation",
            game="sudoku",
            difficulty="easy",
            moves=3,
        )

        result = PuzzleResult(
            type="result",
            success=True,
            state=obs,
        )

        assert result.state is not None
        assert result.state.moves == 3


class TestPuzzleCompletion:
    """Tests for PuzzleCompletion."""

    def test_completion(self):
        """Test completion message."""
        completion = PuzzleCompletion(
            type="complete",
            success=True,
            game="sudoku",
            moves=25,
            invalid_moves=2,
            hints_used=1,
            optimal_steps=20,
        )

        assert completion.type == "complete"
        assert completion.success is True
        assert completion.game == "sudoku"
        assert completion.moves == 25
        assert completion.optimal_steps == 20


class TestPuzzleEpisode:
    """Tests for PuzzleEpisode."""

    def test_basic_episode(self):
        """Test basic episode creation."""
        episode = PuzzleEpisode(
            game="sudoku",
            difficulty="easy",
            seed=42,
            initial_grid=[[0, 1], [2, 0]],
            final_grid=[[3, 1], [2, 4]],
            moves=["place 1 1 3", "place 2 2 4"],
            success=True,
            total_moves=2,
            invalid_moves=0,
            hints_used=0,
            optimal_steps=2,
            difficulty_profile=None,
        )

        assert episode.game == "sudoku"
        assert episode.difficulty == "easy"
        assert episode.seed == 42
        assert len(episode.moves) == 2
        assert episode.success is True


class TestTelnetGymClient:
    """Tests for TelnetGymClient that don't require a connection."""

    def test_client_creation(self):
        """Test client can be created."""
        from chuk_lazarus.data.batching.streaming import TelnetGymClient

        config = TelnetClientConfig()
        client = TelnetGymClient(config)

        assert client.config.host == "localhost"
        assert client.config.port == 8023
        assert client.is_connected is False

    def test_strip_telnet_commands_empty(self):
        """Test stripping telnet commands from empty data."""
        from chuk_lazarus.data.batching.streaming import TelnetGymClient

        client = TelnetGymClient()
        result = client._strip_telnet_commands(b"")
        assert result == b""

    def test_strip_telnet_commands_no_commands(self):
        """Test stripping with no telnet commands."""
        from chuk_lazarus.data.batching.streaming import TelnetGymClient

        client = TelnetGymClient()
        data = b"Hello, World!"
        result = client._strip_telnet_commands(data)
        assert result == data

    def test_strip_telnet_commands_iac(self):
        """Test stripping IAC commands."""
        from chuk_lazarus.data.batching.streaming import TelnetGymClient

        client = TelnetGymClient()
        # IAC DO ECHO (255 253 1)
        data = b"\xff\xfd\x01Hello"
        result = client._strip_telnet_commands(data)
        assert result == b"Hello"

    def test_strip_telnet_commands_multiple(self):
        """Test stripping multiple telnet commands."""
        from chuk_lazarus.data.batching.streaming import TelnetGymClient

        client = TelnetGymClient()
        # IAC WILL ECHO, text, IAC DO LINEMODE
        data = b"\xff\xfb\x01Hello\xff\xfd\x22World"
        result = client._strip_telnet_commands(data)
        assert result == b"HelloWorld"

    def test_strip_telnet_commands_escaped_iac(self):
        """Test stripping escaped IAC (0xFF 0xFF = literal 0xFF)."""
        from chuk_lazarus.data.batching.streaming import TelnetGymClient

        client = TelnetGymClient()
        # Escaped IAC
        data = b"Test\xff\xffData"
        result = client._strip_telnet_commands(data)
        assert result == b"Test\xffData"

    def test_strip_telnet_commands_subnegotiation(self):
        """Test stripping subnegotiation sequences."""
        from chuk_lazarus.data.batching.streaming import TelnetGymClient

        client = TelnetGymClient()
        # IAC SB ... IAC SE
        data = b"\xff\xfa\x01\x00\xff\xf0Hello"
        result = client._strip_telnet_commands(data)
        assert result == b"Hello"


class TestPuzzleEpisodeToTraining:
    """Tests for PuzzleEpisode.to_training_text()."""

    def test_to_training_text(self):
        """Test converting episode to training format."""
        episode = PuzzleEpisode(
            game="sudoku",
            difficulty="easy",
            seed=42,
            initial_grid=[[0, 1], [2, 0]],
            final_grid=[[3, 1], [2, 4]],
            moves=["place 1 1 3", "place 2 2 4"],
            success=True,
            total_moves=2,
            invalid_moves=0,
            hints_used=0,
            optimal_steps=2,
            difficulty_profile=None,
        )

        class MockTokenizer:
            def encode(self, text: str) -> list[int]:
                return list(range(len(text.split())))

        tokenizer = MockTokenizer()
        input_ids, loss_mask = episode.to_training_text(tokenizer)

        # Verify we have input_ids and loss_mask of same length
        assert len(input_ids) == len(loss_mask)

        # Verify loss mask has 0s for prompt and 1s for response
        assert all(m in (0, 1) for m in loss_mask)

        # Should start with 0s (prompt) and end with 1s (response)
        prompt_end = loss_mask.index(1) if 1 in loss_mask else len(loss_mask)
        assert all(m == 0 for m in loss_mask[:prompt_end])
        assert all(m == 1 for m in loss_mask[prompt_end:])

    def test_format_grid(self):
        """Test grid formatting."""
        episode = PuzzleEpisode(
            game="sudoku",
            difficulty="easy",
            seed=42,
            initial_grid=[[1, 0, 3], [0, 5, 0], [7, 8, 9]],
            final_grid=None,
            moves=[],
            success=False,
            total_moves=0,
            invalid_moves=0,
            hints_used=0,
            optimal_steps=None,
            difficulty_profile=None,
        )

        formatted = episode._format_grid(episode.initial_grid)
        lines = formatted.split("\n")

        assert len(lines) == 3
        assert "1" in lines[0]
        assert "." in lines[0]  # 0s become dots
        assert "5" in lines[1]

    def test_format_grid_empty(self):
        """Test formatting empty grid."""
        episode = PuzzleEpisode(
            game="sudoku",
            difficulty="easy",
            seed=42,
            initial_grid=None,
            final_grid=None,
            moves=[],
            success=False,
            total_moves=0,
            invalid_moves=0,
            hints_used=0,
            optimal_steps=None,
            difficulty_profile=None,
        )

        formatted = episode._format_grid(None)
        assert formatted == ""
