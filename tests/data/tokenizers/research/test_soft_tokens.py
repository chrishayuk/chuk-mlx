"""Tests for soft token embeddings."""

import numpy as np
import pytest

from chuk_lazarus.data.tokenizers.research.soft_tokens import (
    InitializationMethod,
    SoftToken,
    SoftTokenBank,
    SoftTokenConfig,
    SoftTokenEmbedding,
    create_control_token,
    create_prompt_tuning_bank,
    create_soft_token,
    initialize_soft_embedding,
    interpolate_embeddings,
)


class TestSoftTokenConfig:
    """Tests for SoftTokenConfig model."""

    def test_default_values(self):
        config = SoftTokenConfig(embedding_dim=768)
        assert config.embedding_dim == 768
        assert config.init_method == InitializationMethod.RANDOM_NORMAL
        assert config.init_std == 0.02
        assert config.trainable is True

    def test_custom_values(self):
        config = SoftTokenConfig(
            embedding_dim=512,
            init_method=InitializationMethod.ZEROS,
            trainable=False,
            learning_rate_multiplier=0.5,
        )
        assert config.embedding_dim == 512
        assert config.init_method == InitializationMethod.ZEROS
        assert config.trainable is False


class TestSoftToken:
    """Tests for SoftToken model."""

    def test_valid_token(self):
        config = SoftTokenConfig(embedding_dim=768)
        token = SoftToken(
            name="test_token",
            token_id=100000,
            description="Test token",
            purpose="testing",
            config=config,
        )
        assert token.name == "test_token"
        assert token.token_id == 100000

    def test_with_created_from(self):
        config = SoftTokenConfig(embedding_dim=768)
        token = SoftToken(
            name="derived",
            token_id=100001,
            config=config,
            created_from="hello world",
        )
        assert token.created_from == "hello world"


class TestSoftTokenEmbedding:
    """Tests for SoftTokenEmbedding model."""

    def test_from_array(self):
        config = SoftTokenConfig(embedding_dim=4)
        token = SoftToken(name="test", token_id=100000, config=config)
        embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        soft_emb = SoftTokenEmbedding.from_array(token, embedding)
        assert len(soft_emb.embedding) == 4
        np.testing.assert_array_almost_equal(soft_emb.embedding_array, embedding)

    def test_embedding_array_property(self):
        config = SoftTokenConfig(embedding_dim=3)
        token = SoftToken(name="test", token_id=100000, config=config)
        soft_emb = SoftTokenEmbedding(token=token, embedding=[1.0, 2.0, 3.0])

        arr = soft_emb.embedding_array
        assert arr.dtype == np.float32
        assert arr.shape == (3,)


class TestSoftTokenBank:
    """Tests for SoftTokenBank model."""

    def test_empty_bank(self):
        bank = SoftTokenBank(name="test_bank", embedding_dim=768)
        assert len(bank.tokens) == 0
        assert bank.next_token_id() == 100000

    def test_add_token(self):
        bank = SoftTokenBank(name="test_bank", embedding_dim=4)
        config = SoftTokenConfig(embedding_dim=4)
        token = SoftToken(name="test", token_id=100000, config=config)
        soft_emb = SoftTokenEmbedding(token=token, embedding=[1.0, 2.0, 3.0, 4.0])

        bank.add_token(soft_emb)
        assert len(bank.tokens) == 1
        assert bank.next_token_id() == 100001

    def test_get_token(self):
        bank = SoftTokenBank(name="test_bank", embedding_dim=4)
        config = SoftTokenConfig(embedding_dim=4)
        token = SoftToken(name="my_token", token_id=100000, config=config)
        soft_emb = SoftTokenEmbedding(token=token, embedding=[1.0, 2.0, 3.0, 4.0])
        bank.add_token(soft_emb)

        found = bank.get_token("my_token")
        assert found is not None
        assert found.token.name == "my_token"

        not_found = bank.get_token("nonexistent")
        assert not_found is None

    def test_get_by_id(self):
        bank = SoftTokenBank(name="test_bank", embedding_dim=4)
        config = SoftTokenConfig(embedding_dim=4)
        token = SoftToken(name="test", token_id=100005, config=config)
        soft_emb = SoftTokenEmbedding(token=token, embedding=[1.0, 2.0, 3.0, 4.0])
        bank.add_token(soft_emb)

        found = bank.get_by_id(100005)
        assert found is not None

        not_found = bank.get_by_id(99999)
        assert not_found is None

    def test_duplicate_id_error(self):
        bank = SoftTokenBank(name="test_bank", embedding_dim=4)
        config = SoftTokenConfig(embedding_dim=4)

        token1 = SoftToken(name="token1", token_id=100000, config=config)
        soft1 = SoftTokenEmbedding(token=token1, embedding=[1.0, 2.0, 3.0, 4.0])
        bank.add_token(soft1)

        token2 = SoftToken(name="token2", token_id=100000, config=config)
        soft2 = SoftTokenEmbedding(token=token2, embedding=[5.0, 6.0, 7.0, 8.0])

        with pytest.raises(ValueError, match="already exists"):
            bank.add_token(soft2)

    def test_get_embeddings_matrix(self):
        bank = SoftTokenBank(name="test_bank", embedding_dim=4)
        config = SoftTokenConfig(embedding_dim=4)

        for i in range(3):
            token = SoftToken(name=f"token_{i}", token_id=100000 + i, config=config)
            emb = [float(i), float(i + 1), float(i + 2), float(i + 3)]
            soft_emb = SoftTokenEmbedding(token=token, embedding=emb)
            bank.add_token(soft_emb)

        matrix = bank.get_embeddings_matrix()
        assert matrix.shape == (3, 4)

    def test_empty_embeddings_matrix(self):
        bank = SoftTokenBank(name="test_bank", embedding_dim=768)
        matrix = bank.get_embeddings_matrix()
        assert matrix.shape == (0, 768)


class TestInitializeSoftEmbedding:
    """Tests for initialize_soft_embedding function."""

    def test_random_normal(self):
        config = SoftTokenConfig(
            embedding_dim=100,
            init_method=InitializationMethod.RANDOM_NORMAL,
            init_mean=0.0,
            init_std=0.02,
        )
        emb = initialize_soft_embedding(config)
        assert emb.shape == (100,)
        assert emb.dtype == np.float32

    def test_random_uniform(self):
        config = SoftTokenConfig(
            embedding_dim=100,
            init_method=InitializationMethod.RANDOM_UNIFORM,
            init_min=-0.1,
            init_max=0.1,
        )
        emb = initialize_soft_embedding(config)
        assert emb.shape == (100,)
        assert np.all(emb >= -0.1)
        assert np.all(emb <= 0.1)

    def test_zeros(self):
        config = SoftTokenConfig(embedding_dim=50, init_method=InitializationMethod.ZEROS)
        emb = initialize_soft_embedding(config)
        np.testing.assert_array_equal(emb, np.zeros(50))

    def test_ones(self):
        config = SoftTokenConfig(embedding_dim=50, init_method=InitializationMethod.ONES)
        emb = initialize_soft_embedding(config)
        np.testing.assert_array_equal(emb, np.ones(50))

    def test_from_tokens(self):
        config = SoftTokenConfig(embedding_dim=4, init_method=InitializationMethod.FROM_TOKENS)
        source = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        emb = initialize_soft_embedding(config, source)
        expected = np.array([3.0, 4.0, 5.0, 6.0])  # Mean
        np.testing.assert_array_almost_equal(emb, expected)

    def test_from_tokens_requires_source(self):
        config = SoftTokenConfig(embedding_dim=4, init_method=InitializationMethod.FROM_TOKENS)
        with pytest.raises(ValueError, match="requires source"):
            initialize_soft_embedding(config)


class TestCreateSoftToken:
    """Tests for create_soft_token function."""

    def test_basic_creation(self):
        config = SoftTokenConfig(embedding_dim=768)
        soft_emb = create_soft_token("my_token", config, token_id=50000)

        assert soft_emb.token.name == "my_token"
        assert soft_emb.token.token_id == 50000
        assert len(soft_emb.embedding) == 768

    def test_with_description(self):
        config = SoftTokenConfig(embedding_dim=768)
        soft_emb = create_soft_token(
            "control_token",
            config,
            description="Controls output style",
            purpose="control",
        )
        assert soft_emb.token.description == "Controls output style"
        assert soft_emb.token.purpose == "control"


class TestInterpolateEmbeddings:
    """Tests for interpolate_embeddings function."""

    def test_linear_interpolation(self):
        e1 = np.array([0.0, 0.0, 0.0])
        e2 = np.array([1.0, 1.0, 1.0])

        result = interpolate_embeddings(e1, e2, alpha=0.5, method="linear")
        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear_endpoints(self):
        e1 = np.array([1.0, 2.0, 3.0])
        e2 = np.array([4.0, 5.0, 6.0])

        result0 = interpolate_embeddings(e1, e2, alpha=0.0, method="linear")
        result1 = interpolate_embeddings(e1, e2, alpha=1.0, method="linear")

        np.testing.assert_array_almost_equal(result0, e1)
        np.testing.assert_array_almost_equal(result1, e2)

    def test_spherical_interpolation(self):
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])

        result = interpolate_embeddings(e1, e2, alpha=0.5, method="spherical")
        # Should be on the unit circle
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01

    def test_invalid_method_raises(self):
        e1 = np.array([1.0, 2.0])
        e2 = np.array([3.0, 4.0])
        with pytest.raises(ValueError, match="Unknown interpolation"):
            interpolate_embeddings(e1, e2, method="invalid")


class TestCreatePromptTuningBank:
    """Tests for create_prompt_tuning_bank function."""

    def test_basic_creation(self):
        bank = create_prompt_tuning_bank(num_tokens=5, embedding_dim=768)
        assert len(bank.tokens) == 5
        assert bank.embedding_dim == 768

    def test_custom_prefix(self):
        bank = create_prompt_tuning_bank(num_tokens=3, embedding_dim=512, prefix="task")
        assert bank.tokens[0].token.name == "task_0"
        assert bank.tokens[1].token.name == "task_1"
        assert bank.tokens[2].token.name == "task_2"

    def test_token_ids_sequential(self):
        bank = create_prompt_tuning_bank(num_tokens=4, embedding_dim=256)
        ids = [t.token.token_id for t in bank.tokens]
        assert ids == [100000, 100001, 100002, 100003]


class TestCreateControlToken:
    """Tests for create_control_token function."""

    def test_basic_control_token(self):
        token = create_control_token(
            "positive_sentiment",
            embedding_dim=768,
            description="Positive sentiment control",
        )
        assert token.token.name == "positive_sentiment"
        assert token.token.purpose == "control"
        assert len(token.embedding) == 768

    def test_from_source_embeddings(self):
        source = np.random.randn(5, 256).astype(np.float32)
        token = create_control_token("derived", embedding_dim=256, source_embeddings=source)
        # Should be initialized from source mean
        expected_mean = source.mean(axis=0)
        np.testing.assert_array_almost_equal(token.embedding_array, expected_mean)
