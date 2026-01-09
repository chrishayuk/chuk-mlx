"""Tests for research_morph command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from chuk_lazarus.cli.commands.tokenizer._types import MorphMethod, ResearchMorphConfig
from chuk_lazarus.cli.commands.tokenizer.research.morph import research_morph


class TestResearchMorphConfig:
    """Tests for ResearchMorphConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.file = "/path/to/embeddings.json"
        args.source = 0
        args.target = 1
        args.method = "linear"
        args.steps = 10
        args.normalize = False
        args.output = None

        config = ResearchMorphConfig.from_args(args)

        assert config.file == Path("/path/to/embeddings.json")
        assert config.source == 0
        assert config.target == 1
        assert config.method == MorphMethod.LINEAR
        assert config.steps == 10
        assert config.normalize is False
        assert config.output is None

    def test_from_args_with_options(self):
        """Test config with all options."""
        args = MagicMock()
        args.file = "/path/to/embeddings.json"
        args.source = 5
        args.target = 10
        args.method = "slerp"
        args.steps = 20
        args.normalize = True
        args.output = Path("/path/to/trajectory.json")

        config = ResearchMorphConfig.from_args(args)

        assert config.source == 5
        assert config.target == 10
        assert config.method == MorphMethod.SLERP
        assert config.steps == 20
        assert config.normalize is True
        assert config.output == Path("/path/to/trajectory.json")


class TestResearchMorph:
    """Tests for research_morph function."""

    @patch("chuk_lazarus.data.tokenizers.research.morph_token")
    @patch("chuk_lazarus.data.tokenizers.research.compute_path_length")
    @patch("chuk_lazarus.data.tokenizers.research.compute_straightness")
    @patch("builtins.open")
    @patch("json.load")
    def test_morph_basic(
        self, mock_json_load, mock_open, mock_straightness, mock_path_length, mock_morph, capsys
    ):
        """Test basic token morphing."""
        mock_json_load.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "token_strs": ["hello", "world"],
        }

        mock_result = MagicMock()
        mock_result.source_token = "hello"
        mock_result.target_token = "world"
        mock_result.method = MorphMethod.LINEAR
        mock_result.num_steps = 10
        mock_result.alphas = [0.0, 0.5, 1.0]
        mock_result.get_embeddings_array.return_value = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.25, 0.35, 0.45],
                [0.4, 0.5, 0.6],
            ]
        )
        mock_morph.return_value = mock_result
        mock_path_length.return_value = 0.5
        mock_straightness.return_value = 0.99

        config = ResearchMorphConfig(
            file=Path("/tmp/embeddings.json"),
            source=0,
            target=1,
            method=MorphMethod.LINEAR,
            steps=10,
        )
        research_morph(config)

        captured = capsys.readouterr()
        assert "Token Morphing" in captured.out
        assert "Source:" in captured.out
        assert "Target:" in captured.out
        assert "Method:" in captured.out
        assert "Path length:" in captured.out
        assert "Straightness:" in captured.out
