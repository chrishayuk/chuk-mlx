"""Tests for token morphing and blending."""

import numpy as np
import pytest

from chuk_lazarus.data.tokenizers.research.token_morphing import (
    BlendMode,
    MorphConfig,
    MorphMethod,
    MorphResult,
    MorphSequence,
    TokenBlend,
    blend_tokens,
    compute_path_length,
    compute_straightness,
    create_morph_sequence,
    find_midpoint,
    morph_token,
)


class TestMorphConfig:
    """Tests for MorphConfig model."""

    def test_default_values(self):
        config = MorphConfig()
        assert config.method == MorphMethod.LINEAR
        assert config.num_steps == 10
        assert config.include_endpoints is True

    def test_custom_values(self):
        config = MorphConfig(
            method=MorphMethod.SPHERICAL,
            num_steps=20,
            normalize_output=True,
        )
        assert config.method == MorphMethod.SPHERICAL
        assert config.num_steps == 20


class TestMorphResult:
    """Tests for MorphResult model."""

    def test_get_embeddings_array(self):
        result = MorphResult(
            source_token="a",
            target_token="b",
            method=MorphMethod.LINEAR,
            num_steps=3,
            alphas=[0.0, 0.5, 1.0],
            embeddings=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        )
        arr = result.get_embeddings_array()
        assert arr.shape == (3, 2)

    def test_get_embedding_at(self):
        result = MorphResult(
            source_token="a",
            target_token="b",
            method=MorphMethod.LINEAR,
            num_steps=3,
            alphas=[0.0, 0.5, 1.0],
            embeddings=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        )
        emb = result.get_embedding_at(0.5)
        np.testing.assert_array_almost_equal(emb, [0.5, 0.5])


class TestMorphSequence:
    """Tests for MorphSequence model."""

    def test_get_full_sequence(self):
        morph1 = MorphResult(
            source_token="a",
            target_token="b",
            method=MorphMethod.LINEAR,
            num_steps=3,
            alphas=[0.0, 0.5, 1.0],
            embeddings=[[0.0], [0.5], [1.0]],
        )
        morph2 = MorphResult(
            source_token="b",
            target_token="c",
            method=MorphMethod.LINEAR,
            num_steps=3,
            alphas=[0.0, 0.5, 1.0],
            embeddings=[[1.0], [1.5], [2.0]],
        )
        seq = MorphSequence(tokens=["a", "b", "c"], morphs=[morph1, morph2], total_steps=5)
        full = seq.get_full_sequence()
        # Should be [0, 0.5, 1, 1.5, 2] (removing duplicate 1.0)
        assert full.shape == (5, 1)


class TestTokenBlend:
    """Tests for TokenBlend model."""

    def test_get_embedding_array(self):
        blend = TokenBlend(
            tokens=["a", "b"],
            weights=[0.5, 0.5],
            mode=BlendMode.AVERAGE,
            embedding=[0.5, 0.5, 0.5],
        )
        arr = blend.get_embedding_array()
        assert arr.shape == (3,)


class TestMorphToken:
    """Tests for morph_token function."""

    def test_linear_morph(self):
        e1 = np.array([0.0, 0.0, 0.0])
        e2 = np.array([1.0, 1.0, 1.0])

        result = morph_token(e1, e2, "start", "end")
        assert result.source_token == "start"
        assert result.target_token == "end"
        assert len(result.embeddings) == 10  # Default num_steps

    def test_morph_endpoints(self):
        e1 = np.array([0.0, 0.0])
        e2 = np.array([1.0, 1.0])

        config = MorphConfig(num_steps=3, include_endpoints=True)
        result = morph_token(e1, e2, "a", "b", config)

        # First should be e1, last should be e2
        np.testing.assert_array_almost_equal(result.embeddings[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(result.embeddings[-1], [1.0, 1.0])

    def test_morph_exclude_endpoints(self):
        e1 = np.array([0.0, 0.0])
        e2 = np.array([1.0, 1.0])

        config = MorphConfig(num_steps=2, include_endpoints=False)
        result = morph_token(e1, e2, "a", "b", config)

        # Should not include exact endpoints
        assert not np.allclose(result.embeddings[0], [0.0, 0.0])
        assert not np.allclose(result.embeddings[-1], [1.0, 1.0])

    def test_spherical_morph(self):
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])

        config = MorphConfig(method=MorphMethod.SPHERICAL, num_steps=5)
        result = morph_token(e1, e2, "a", "b", config)

        # All intermediate points should have similar norm
        embeddings = result.get_embeddings_array()
        norms = np.linalg.norm(embeddings, axis=1)
        # Should be close to 1.0 for unit vectors
        assert np.std(norms) < 0.1

    def test_bezier_morph(self):
        e1 = np.array([0.0, 0.0])
        e2 = np.array([1.0, 1.0])

        config = MorphConfig(method=MorphMethod.BEZIER, num_steps=5)
        result = morph_token(e1, e2, "a", "b", config)

        assert len(result.embeddings) == 5
        assert result.method == MorphMethod.BEZIER

    def test_cubic_morph(self):
        e1 = np.array([0.0, 0.0])
        e2 = np.array([1.0, 1.0])

        config = MorphConfig(method=MorphMethod.CUBIC, num_steps=5)
        result = morph_token(e1, e2, "a", "b", config)

        assert len(result.embeddings) == 5

    def test_normalized_output(self):
        e1 = np.array([2.0, 0.0])
        e2 = np.array([0.0, 3.0])

        config = MorphConfig(num_steps=5, normalize_output=True)
        result = morph_token(e1, e2, "a", "b", config)

        embeddings = result.get_embeddings_array()
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(5))


class TestCreateMorphSequence:
    """Tests for create_morph_sequence function."""

    def test_basic_sequence(self):
        embeddings = [np.array([0.0]), np.array([1.0]), np.array([2.0])]
        names = ["a", "b", "c"]

        seq = create_morph_sequence(embeddings, names)
        assert seq.tokens == ["a", "b", "c"]
        assert len(seq.morphs) == 2

    def test_too_few_tokens_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            create_morph_sequence([np.array([0.0])], ["a"])

    def test_mismatched_lengths_error(self):
        with pytest.raises(ValueError, match="same length"):
            create_morph_sequence([np.array([0.0]), np.array([1.0])], ["a"])


class TestBlendTokens:
    """Tests for blend_tokens function."""

    def test_average_blend(self):
        embeddings = [np.array([0.0, 0.0]), np.array([2.0, 2.0])]
        names = ["a", "b"]

        result = blend_tokens(embeddings, names, mode=BlendMode.AVERAGE)
        expected = np.array([1.0, 1.0])
        np.testing.assert_array_almost_equal(result.get_embedding_array(), expected)

    def test_weighted_blend(self):
        embeddings = [np.array([0.0, 0.0]), np.array([4.0, 4.0])]
        names = ["a", "b"]
        weights = [0.75, 0.25]

        result = blend_tokens(embeddings, names, weights=weights, mode=BlendMode.WEIGHTED)
        expected = np.array([1.0, 1.0])  # 0.75 * 0 + 0.25 * 4 = 1.0
        np.testing.assert_array_almost_equal(result.get_embedding_array(), expected)

    def test_geometric_blend(self):
        embeddings = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
        names = ["a", "b"]

        result = blend_tokens(embeddings, names, mode=BlendMode.GEOMETRIC)
        # Geometric mean of identical vectors
        assert result.mode == BlendMode.GEOMETRIC

    def test_attention_blend(self):
        embeddings = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        names = ["a", "b", "c"]

        result = blend_tokens(embeddings, names, mode=BlendMode.ATTENTION)
        assert result.mode == BlendMode.ATTENTION
        assert len(result.embedding) == 2

    def test_empty_error(self):
        with pytest.raises(ValueError, match="at least 1"):
            blend_tokens([], [])

    def test_mismatched_weights_error(self):
        with pytest.raises(ValueError, match="must match"):
            blend_tokens(
                [np.array([1.0]), np.array([2.0])],
                ["a", "b"],
                weights=[0.5],  # Wrong length
            )


class TestFindMidpoint:
    """Tests for find_midpoint function."""

    def test_linear_midpoint(self):
        e1 = np.array([0.0, 0.0])
        e2 = np.array([2.0, 2.0])

        mid = find_midpoint(e1, e2, method=MorphMethod.LINEAR)
        np.testing.assert_array_almost_equal(mid, [1.0, 1.0])

    def test_spherical_midpoint(self):
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])

        mid = find_midpoint(e1, e2, method=MorphMethod.SPHERICAL)
        # Midpoint should be on the arc
        norm = np.linalg.norm(mid)
        assert abs(norm - 1.0) < 0.01


class TestComputePathLength:
    """Tests for compute_path_length function."""

    def test_straight_path(self):
        result = MorphResult(
            source_token="a",
            target_token="b",
            method=MorphMethod.LINEAR,
            num_steps=3,
            alphas=[0.0, 0.5, 1.0],
            embeddings=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        )
        length = compute_path_length(result)
        assert abs(length - 1.0) < 0.01

    def test_empty_path(self):
        result = MorphResult(
            source_token="a",
            target_token="b",
            method=MorphMethod.LINEAR,
            num_steps=1,
            alphas=[0.0],
            embeddings=[[0.0, 0.0]],
        )
        length = compute_path_length(result)
        assert length == 0.0


class TestComputeStraightness:
    """Tests for compute_straightness function."""

    def test_perfectly_straight(self):
        result = MorphResult(
            source_token="a",
            target_token="b",
            method=MorphMethod.LINEAR,
            num_steps=5,
            alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
            embeddings=[
                [0.0, 0.0],
                [0.25, 0.0],
                [0.5, 0.0],
                [0.75, 0.0],
                [1.0, 0.0],
            ],
        )
        straightness = compute_straightness(result)
        assert abs(straightness - 1.0) < 0.01

    def test_curved_path(self):
        # A curved path should have straightness < 1
        result = MorphResult(
            source_token="a",
            target_token="b",
            method=MorphMethod.LINEAR,
            num_steps=3,
            alphas=[0.0, 0.5, 1.0],
            embeddings=[
                [0.0, 0.0],
                [0.5, 1.0],  # Curve up
                [1.0, 0.0],
            ],
        )
        straightness = compute_straightness(result)
        assert straightness < 1.0
