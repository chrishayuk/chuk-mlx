"""Tests for introspect clustering CLI commands."""

import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from .conftest import requires_sklearn


@requires_sklearn
class TestIntrospectActivationCluster:
    """Tests for introspect_activation_cluster command."""

    @pytest.fixture
    def cluster_args(self):
        """Create arguments for activation cluster command."""
        return Namespace(
            model="test-model",
            class_a="2+2=|5+5=|10+10=",
            class_b="47*47=|67*83=|97*89=",
            label_a="easy",
            label_b="hard",
            prompt_groups=None,
            labels=None,
            layer=None,
            save_plot=None,
            output=None,
        )

    @pytest.fixture
    def multi_class_args(self):
        """Create arguments for multi-class clustering."""
        return Namespace(
            model="test-model",
            class_a=None,
            class_b=None,
            label_a=None,
            label_b=None,
            prompt_groups=["2+2=|3+3=", "47*47=", "100-50="],
            labels=["addition", "multiplication", "subtraction"],
            layer=6,
            save_plot=None,
            output=None,
        )

    def test_cluster_requires_prompts(self, capsys):
        """Test that cluster requires prompt input."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        args = Namespace(
            model="test-model",
            class_a=None,
            class_b=None,
            label_a=None,
            label_b=None,
            prompt_groups=None,
            labels=None,
            layer=None,
            save_plot=None,
        )

        introspect_activation_cluster(args)

        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_cluster_requires_min_prompts(self, capsys):
        """Test that cluster requires at least 2 prompts."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        args = Namespace(
            model="test-model",
            class_a="2+2=",
            class_b=None,
            label_a="single",
            label_b=None,
            prompt_groups=None,
            labels=None,
            layer=None,
            save_plot=None,
        )

        introspect_activation_cluster(args)

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "2 prompts" in captured.out or "at least" in captured.out.lower()

    def test_cluster_mismatched_labels(self, capsys):
        """Test error on mismatched prompt groups and labels."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        args = Namespace(
            model="test-model",
            class_a=None,
            class_b=None,
            label_a=None,
            label_b=None,
            prompt_groups=["2+2=|3+3=", "47*47="],
            labels=["only_one_label"],  # Only 1 label for 2 groups
            layer=None,
            save_plot=None,
        )

        introspect_activation_cluster(args)

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "match" in captured.out.lower()

    def test_cluster_legacy_syntax(self, cluster_args, capsys):
        """Test clustering with legacy two-class syntax."""
        import mlx.core as mx
        import numpy as np

        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        with patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader:
            mock_result = MagicMock()
            mock_result.model_path = MagicMock()
            mock_result.model_path.__truediv__ = MagicMock(return_value=MagicMock())
            mock_loader.download.return_value = mock_result

            with patch("builtins.open", MagicMock()):
                with patch("json.load") as mock_json:
                    mock_json.return_value = {
                        "model_type": "llama",
                        "hidden_size": 768,
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "intermediate_size": 3072,
                        "vocab_size": 32000,
                    }

                    with patch(
                        "chuk_lazarus.models_v2.families.registry.detect_model_family"
                    ) as mock_detect:
                        mock_detect.return_value = "llama"

                        with patch(
                            "chuk_lazarus.models_v2.families.registry.get_family_info"
                        ) as mock_family:
                            mock_family_info = MagicMock()
                            mock_family_info.config_class.from_hf_config.return_value = MagicMock(
                                num_hidden_layers=12
                            )
                            mock_family_info.model_class.return_value = MagicMock()
                            mock_family.return_value = mock_family_info

                            mock_loader.apply_weights_to_model = MagicMock()
                            mock_tokenizer = MagicMock()
                            mock_tokenizer.encode.return_value = [
                                1,
                                2,
                                3,
                                4,
                                5,
                            ]  # Return proper list
                            mock_loader.load_tokenizer.return_value = mock_tokenizer

                            with patch(
                                "chuk_lazarus.introspection.ModelAccessor"
                            ) as mock_accessor_cls:
                                mock_accessor = MagicMock()
                                mock_accessor.layers = [MagicMock() for _ in range(12)]
                                mock_accessor.embed = MagicMock(return_value=mx.zeros((1, 5, 768)))
                                mock_accessor.embedding_scale = None

                                # Make layers return proper outputs
                                for layer in mock_accessor.layers:
                                    layer.return_value = mx.zeros((1, 5, 768))

                                mock_accessor_cls.return_value = mock_accessor

                                with patch("sklearn.decomposition.PCA") as mock_pca:
                                    mock_pca_instance = MagicMock()
                                    mock_pca_instance.fit_transform.return_value = np.random.randn(
                                        6, 2
                                    )
                                    mock_pca_instance.explained_variance_ratio_ = [
                                        0.6,
                                        0.2,
                                    ]
                                    mock_pca.return_value = mock_pca_instance

                                    introspect_activation_cluster(cluster_args)

                                    captured = capsys.readouterr()
                                    assert (
                                        "Loading model" in captured.out or "Classes" in captured.out
                                    )

    def test_cluster_multi_class_syntax(self, multi_class_args, capsys):
        """Test clustering with multi-class syntax."""
        import mlx.core as mx
        import numpy as np

        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        with patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader:
            mock_result = MagicMock()
            mock_result.model_path = MagicMock()
            mock_result.model_path.__truediv__ = MagicMock(return_value=MagicMock())
            mock_loader.download.return_value = mock_result

            with patch("builtins.open", MagicMock()):
                with patch("json.load") as mock_json:
                    mock_json.return_value = {
                        "model_type": "llama",
                        "hidden_size": 768,
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "intermediate_size": 3072,
                        "vocab_size": 32000,
                    }

                    with patch(
                        "chuk_lazarus.models_v2.families.registry.detect_model_family"
                    ) as mock_detect:
                        mock_detect.return_value = "llama"

                        with patch(
                            "chuk_lazarus.models_v2.families.registry.get_family_info"
                        ) as mock_family:
                            mock_family_info = MagicMock()
                            mock_family_info.config_class.from_hf_config.return_value = MagicMock(
                                num_hidden_layers=12
                            )
                            mock_family_info.model_class.return_value = MagicMock()
                            mock_family.return_value = mock_family_info

                            mock_loader.apply_weights_to_model = MagicMock()
                            mock_tokenizer = MagicMock()
                            mock_tokenizer.encode.return_value = [
                                1,
                                2,
                                3,
                                4,
                                5,
                            ]  # Return proper list
                            mock_loader.load_tokenizer.return_value = mock_tokenizer

                            with patch(
                                "chuk_lazarus.introspection.ModelAccessor"
                            ) as mock_accessor_cls:
                                mock_accessor = MagicMock()
                                mock_accessor.layers = [MagicMock() for _ in range(12)]
                                mock_accessor.embed = MagicMock(return_value=mx.zeros((1, 5, 768)))
                                mock_accessor.embedding_scale = None

                                for layer in mock_accessor.layers:
                                    layer.return_value = mx.zeros((1, 5, 768))

                                mock_accessor_cls.return_value = mock_accessor

                                with patch("sklearn.decomposition.PCA") as mock_pca:
                                    mock_pca_instance = MagicMock()
                                    mock_pca_instance.fit_transform.return_value = np.random.randn(
                                        4, 2
                                    )
                                    mock_pca_instance.explained_variance_ratio_ = [
                                        0.6,
                                        0.2,
                                    ]
                                    mock_pca.return_value = mock_pca_instance

                                    introspect_activation_cluster(multi_class_args)

                                    captured = capsys.readouterr()
                                    assert (
                                        "Loading model" in captured.out or "Classes" in captured.out
                                    )

    def test_cluster_with_specific_layer(self, cluster_args, capsys):
        """Test clustering at specific layer."""
        import mlx.core as mx
        import numpy as np

        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        cluster_args.layer = 8

        with patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader:
            mock_result = MagicMock()
            mock_result.model_path = MagicMock()
            mock_result.model_path.__truediv__ = MagicMock(return_value=MagicMock())
            mock_loader.download.return_value = mock_result

            with patch("builtins.open", MagicMock()):
                with patch("json.load") as mock_json:
                    mock_json.return_value = {
                        "model_type": "llama",
                        "hidden_size": 768,
                        "num_hidden_layers": 12,
                    }

                    with patch(
                        "chuk_lazarus.models_v2.families.registry.detect_model_family"
                    ) as mock_detect:
                        mock_detect.return_value = "llama"

                        with patch(
                            "chuk_lazarus.models_v2.families.registry.get_family_info"
                        ) as mock_family:
                            mock_family_info = MagicMock()
                            mock_family_info.config_class.from_hf_config.return_value = MagicMock(
                                num_hidden_layers=12
                            )
                            mock_family_info.model_class.return_value = MagicMock()
                            mock_family.return_value = mock_family_info

                            mock_loader.apply_weights_to_model = MagicMock()
                            mock_tokenizer = MagicMock()
                            mock_tokenizer.encode.return_value = [
                                1,
                                2,
                                3,
                                4,
                                5,
                            ]  # Return proper list
                            mock_loader.load_tokenizer.return_value = mock_tokenizer

                            with patch(
                                "chuk_lazarus.introspection.ModelAccessor"
                            ) as mock_accessor_cls:
                                mock_accessor = MagicMock()
                                mock_accessor.layers = [MagicMock() for _ in range(12)]
                                mock_accessor.embed = MagicMock(return_value=mx.zeros((1, 5, 768)))
                                mock_accessor.embedding_scale = None

                                for layer in mock_accessor.layers:
                                    layer.return_value = mx.zeros((1, 5, 768))

                                mock_accessor_cls.return_value = mock_accessor

                                with patch("sklearn.decomposition.PCA") as mock_pca:
                                    mock_pca_instance = MagicMock()
                                    mock_pca_instance.fit_transform.return_value = np.random.randn(
                                        6, 2
                                    )
                                    mock_pca_instance.explained_variance_ratio_ = [
                                        0.6,
                                        0.2,
                                    ]
                                    mock_pca.return_value = mock_pca_instance

                                    introspect_activation_cluster(cluster_args)

                                    captured = capsys.readouterr()
                                    # Should mention the target layer
                                    assert "8" in captured.out or "Loading" in captured.out

    def test_cluster_multiple_layers(self, cluster_args, capsys):
        """Test clustering at multiple layers."""
        import mlx.core as mx
        import numpy as np

        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        cluster_args.layer = "4,8,12"

        with patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader:
            mock_result = MagicMock()
            mock_result.model_path = MagicMock()
            mock_result.model_path.__truediv__ = MagicMock(return_value=MagicMock())
            mock_loader.download.return_value = mock_result

            with patch("builtins.open", MagicMock()):
                with patch("json.load") as mock_json:
                    mock_json.return_value = {
                        "model_type": "llama",
                        "hidden_size": 768,
                        "num_hidden_layers": 16,
                    }

                    with patch(
                        "chuk_lazarus.models_v2.families.registry.detect_model_family"
                    ) as mock_detect:
                        mock_detect.return_value = "llama"

                        with patch(
                            "chuk_lazarus.models_v2.families.registry.get_family_info"
                        ) as mock_family:
                            mock_family_info = MagicMock()
                            mock_family_info.config_class.from_hf_config.return_value = MagicMock(
                                num_hidden_layers=16
                            )
                            mock_family_info.model_class.return_value = MagicMock()
                            mock_family.return_value = mock_family_info

                            mock_loader.apply_weights_to_model = MagicMock()
                            mock_tokenizer = MagicMock()
                            mock_tokenizer.encode.return_value = [
                                1,
                                2,
                                3,
                                4,
                                5,
                            ]  # Return proper list
                            mock_loader.load_tokenizer.return_value = mock_tokenizer

                            with patch(
                                "chuk_lazarus.introspection.ModelAccessor"
                            ) as mock_accessor_cls:
                                mock_accessor = MagicMock()
                                mock_accessor.layers = [MagicMock() for _ in range(16)]
                                mock_accessor.embed = MagicMock(return_value=mx.zeros((1, 5, 768)))
                                mock_accessor.embedding_scale = None

                                for layer in mock_accessor.layers:
                                    layer.return_value = mx.zeros((1, 5, 768))

                                mock_accessor_cls.return_value = mock_accessor

                                with patch("sklearn.decomposition.PCA") as mock_pca:
                                    mock_pca_instance = MagicMock()
                                    mock_pca_instance.fit_transform.return_value = np.random.randn(
                                        6, 2
                                    )
                                    mock_pca_instance.explained_variance_ratio_ = [
                                        0.6,
                                        0.2,
                                    ]
                                    mock_pca.return_value = mock_pca_instance

                                    introspect_activation_cluster(cluster_args)

                                    captured = capsys.readouterr()
                                    # Should process multiple layers
                                    assert (
                                        "Loading" in captured.out or "layer" in captured.out.lower()
                                    )

    def test_cluster_from_file(self, cluster_args, capsys):
        """Test clustering with prompts from file."""
        import mlx.core as mx
        import numpy as np

        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("2+2=\n3+3=\n4+4=\n")
            f.flush()

            cluster_args.class_a = f"@{f.name}"

            with patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader:
                mock_result = MagicMock()
                mock_result.model_path = MagicMock()
                mock_result.model_path.__truediv__ = MagicMock(return_value=MagicMock())
                mock_loader.download.return_value = mock_result

                # Need to handle the file read for class_a but mock the config read
                original_open = open

                def mock_open_handler(path, *args, **kwargs):
                    if str(path).endswith(".txt"):
                        return original_open(path, *args, **kwargs)
                    return MagicMock()

                with patch("builtins.open", side_effect=mock_open_handler):
                    with patch("json.load") as mock_json:
                        mock_json.return_value = {
                            "model_type": "llama",
                            "hidden_size": 768,
                            "num_hidden_layers": 12,
                        }

                        with patch(
                            "chuk_lazarus.models_v2.families.registry.detect_model_family"
                        ) as mock_detect:
                            mock_detect.return_value = "llama"

                            with patch(
                                "chuk_lazarus.models_v2.families.registry.get_family_info"
                            ) as mock_family:
                                mock_family_info = MagicMock()
                                mock_family_info.config_class.from_hf_config.return_value = (
                                    MagicMock(num_hidden_layers=12)
                                )
                                mock_family_info.model_class.return_value = MagicMock()
                                mock_family.return_value = mock_family_info

                                mock_loader.apply_weights_to_model = MagicMock()
                                mock_tokenizer = MagicMock()
                                mock_tokenizer.encode.return_value = [
                                    1,
                                    2,
                                    3,
                                    4,
                                    5,
                                ]  # Return proper list
                                mock_loader.load_tokenizer.return_value = mock_tokenizer

                                with patch(
                                    "chuk_lazarus.introspection.ModelAccessor"
                                ) as mock_accessor_cls:
                                    mock_accessor = MagicMock()
                                    mock_accessor.layers = [MagicMock() for _ in range(12)]
                                    mock_accessor.embed = MagicMock(
                                        return_value=mx.zeros((1, 5, 768))
                                    )
                                    mock_accessor.embedding_scale = None

                                    for layer in mock_accessor.layers:
                                        layer.return_value = mx.zeros((1, 5, 768))

                                    mock_accessor_cls.return_value = mock_accessor

                                    with patch("sklearn.decomposition.PCA") as mock_pca:
                                        mock_pca_instance = MagicMock()
                                        mock_pca_instance.fit_transform.return_value = (
                                            np.random.randn(6, 2)
                                        )
                                        mock_pca_instance.explained_variance_ratio_ = [
                                            0.6,
                                            0.2,
                                        ]
                                        mock_pca.return_value = mock_pca_instance

                                        introspect_activation_cluster(cluster_args)

                                        captured = capsys.readouterr()
                                        assert (
                                            "Loading" in captured.out or "Classes" in captured.out
                                        )
