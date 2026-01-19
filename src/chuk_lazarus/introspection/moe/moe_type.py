"""MoE type detection: native vs upcycled classification.

Determines whether an MoE model was trained natively as MoE or converted from
a dense model (upcycled). This affects compression strategies:

- Native-MoE: Experts are orthogonal. Not compressible via SVD (need quantization/pruning).
- Upcycled-MoE: Experts share a base with low-rank deltas. Potentially compressible via SVD overlay.

Key metrics:
- Gate rank: Native-MoE has high rank (>50%), Upcycled would have low rank (<5%)
- Cosine similarity: Native-MoE has ~0 (orthogonal), Upcycled would have >0.3 (clustered)

Note: Both GPT-OSS and OLMoE show native MoE characteristics (high gate rank, low similarity).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .detector import _get_layers, get_moe_layers
from .enums import MoEArchitecture, MoEType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn


# =============================================================================
# Pydantic Models
# =============================================================================


class ProjectionRankAnalysis(BaseModel):
    """SVD rank analysis for a single projection type (gate, up, or down)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Projection name: gate, up, or down")
    shape: tuple[int, int] = Field(description="(out_features, in_features)")
    max_rank: int = Field(ge=1, description="Maximum possible rank (min of dimensions)")
    effective_rank_95: int = Field(ge=0, description="Rank capturing 95% variance")

    @property
    def rank_ratio(self) -> float:
        """Fraction of max rank used (0.0 to 1.0)."""
        return self.effective_rank_95 / self.max_rank if self.max_rank > 0 else 0.0

    @property
    def compression_ratio(self) -> float:
        """Potential compression from SVD factorization."""
        if self.effective_rank_95 == 0:
            return float("inf")
        m, n = self.shape
        original = m * n
        factorized = self.effective_rank_95 * (m + n)
        return original / factorized


class TrainingOriginSignals(BaseModel):
    """Signals indicating how the MoE model was trained."""

    model_config = ConfigDict(frozen=True)

    # Primary signals
    expert_similarity: float = Field(
        ge=-1.0,
        le=1.0,
        description="Mean cosine similarity (high=upcycled, low=pretrained)",
    )
    rank_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Effective rank / max rank (low=upcycled, high=pretrained)",
    )

    # Additional signals
    layer_consistency: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How consistent the pattern is across layers (1.0=same across all layers)",
    )
    expert_norm_variance: float = Field(
        default=0.0,
        ge=0.0,
        description="Variance in expert weight norms (low=upcycled, high=pretrained)",
    )
    router_entropy: float = Field(
        default=0.0,
        ge=0.0,
        description="Router weight entropy (low=specialized, high=uniform)",
    )

    @property
    def upcycled_score(self) -> float:
        """Score indicating likelihood of upcycling (0-1)."""
        # High similarity + low rank = upcycled
        sim_score = max(0, min(1, (self.expert_similarity - 0.1) / 0.4))  # 0.1->0, 0.5->1
        rank_score = max(0, min(1, 1 - (self.rank_ratio / 0.2)))  # 0->1, 0.2->0
        consistency_score = self.layer_consistency
        norm_score = max(0, min(1, 1 - self.expert_norm_variance * 10))  # Low variance = upcycled

        # Weighted combination
        return 0.4 * sim_score + 0.3 * rank_score + 0.2 * consistency_score + 0.1 * norm_score

    @property
    def pretrained_score(self) -> float:
        """Score indicating likelihood of pretraining as MoE (0-1)."""
        # Low similarity + high rank = pretrained
        sim_score = max(0, min(1, (0.1 - self.expert_similarity) / 0.1))  # 0.1->0, 0->1
        rank_score = max(0, min(1, (self.rank_ratio - 0.3) / 0.3))  # 0.3->0, 0.6->1

        # Weighted combination
        return 0.5 * sim_score + 0.5 * rank_score


class MoETypeAnalysis(BaseModel):
    """Complete MoE type analysis result."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Model identifier")
    layer_idx: int = Field(ge=0, description="Analyzed layer index")
    num_experts: int = Field(ge=1, description="Number of experts")
    moe_type: MoEType = Field(description="Detected MoE type (upcycled/pretrained_moe/unknown)")
    architecture: MoEArchitecture = Field(
        default=MoEArchitecture.GENERIC,
        description="Detected MoE architecture",
    )

    # Projection analysis
    gate: ProjectionRankAnalysis = Field(description="Gate projection analysis")
    up: ProjectionRankAnalysis = Field(description="Up projection analysis")
    down: ProjectionRankAnalysis = Field(description="Down projection analysis")

    # Similarity metrics
    mean_cosine_similarity: float = Field(
        ge=-1.0,
        le=1.0,
        description="Mean pairwise expert cosine similarity",
    )
    std_cosine_similarity: float = Field(
        ge=0.0,
        description="Std dev of pairwise cosine similarities",
    )

    # Training origin detection
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the classification (0-1)",
    )
    training_signals: TrainingOriginSignals | None = Field(
        default=None,
        description="Detailed signals for training origin detection",
    )

    # Optional detailed similarity data (for visualization)
    similarity_matrix: tuple[tuple[float, ...], ...] | None = Field(
        default=None,
        description="Full pairwise similarity matrix (num_experts x num_experts)",
    )

    @property
    def estimated_compression(self) -> float:
        """Overall compression estimate based on all projections."""
        total_original = sum(p.shape[0] * p.shape[1] for p in [self.gate, self.up, self.down])
        total_compressed = sum(
            p.effective_rank_95 * (p.shape[0] + p.shape[1]) for p in [self.gate, self.up, self.down]
        )
        return total_original / total_compressed if total_compressed > 0 else 1.0

    @property
    def is_compressible(self) -> bool:
        """Whether model benefits from SVD compression (upcycled MoE only)."""
        return self.moe_type == MoEType.UPCYCLED

    @property
    def training_origin(self) -> str:
        """Human-readable training origin description."""
        if self.moe_type == MoEType.UPCYCLED:
            return "Upcycled from dense model"
        elif self.moe_type == MoEType.PRETRAINED_MOE:
            return "Pretrained as MoE from scratch"
        else:
            return "Unknown training origin"

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence level."""
        if self.confidence >= 0.8:
            return "High"
        elif self.confidence >= 0.5:
            return "Medium"
        elif self.confidence >= 0.3:
            return "Low"
        else:
            return "Very Low"


# =============================================================================
# Service Class
# =============================================================================


class MoETypeService:
    """Service for detecting MoE type (pseudo vs native).

    Usage:
        result = await MoETypeService.analyze("openai/gpt-oss-20b")
        print(result.moe_type)  # MoEType.PSEUDO
    """

    # Classification thresholds
    PSEUDO_RANK_RATIO_THRESHOLD: float = 0.05
    """Gate rank ratio below this suggests pseudo-MoE."""

    PSEUDO_SIMILARITY_THRESHOLD: float = 0.25
    """Cosine similarity above this suggests pseudo-MoE."""

    NATIVE_RANK_RATIO_THRESHOLD: float = 0.50
    """Gate rank ratio above this suggests native-MoE."""

    NATIVE_SIMILARITY_THRESHOLD: float = 0.10
    """Cosine similarity below this suggests native-MoE."""

    VARIANCE_THRESHOLD: float = 0.95
    """Variance threshold for effective rank computation."""

    MAX_EXPERTS_TO_SAMPLE: int = 8
    """Maximum experts to sample for SVD (full analysis is expensive)."""

    @classmethod
    async def analyze(
        cls,
        model_id: str,
        *,
        layer: int | None = None,
    ) -> MoETypeAnalysis:
        """Analyze a model to detect MoE type.

        Args:
            model_id: HuggingFace model ID or local path
            layer: Specific layer to analyze (default: first MoE layer)

        Returns:
            MoETypeAnalysis with classification and evidence

        Raises:
            ValueError: If model has no MoE layers
        """
        return await asyncio.to_thread(cls._analyze_sync, model_id, layer)

    @classmethod
    def _analyze_sync(
        cls,
        model_id: str,
        layer: int | None,
    ) -> MoETypeAnalysis:
        """Synchronous implementation of analysis."""

        # Load model using lazarus loader (handles all architectures)
        model = cls._load_model(model_id)

        # Find MoE layers
        moe_layers = get_moe_layers(model)
        if not moe_layers:
            raise ValueError(f"No MoE layers found in {model_id}")

        layer_idx = layer if layer is not None else moe_layers[0]

        # Detect architecture
        from .detector import detect_moe_architecture

        architecture = detect_moe_architecture(model)

        # Get experts for this layer
        experts = cls._get_experts(model, layer_idx)
        if experts is None:
            raise ValueError(f"Could not extract experts from layer {layer_idx}")

        # Extract weights based on architecture
        logger.info("Extracting expert weights...")
        gate_w, up_w, down_w, num_experts = cls._extract_weights(experts, architecture)
        logger.info(f"Found {num_experts} experts, weight shape: {tuple(gate_w.shape)}")

        # Analyze each projection
        logger.info("Analyzing gate projection...")
        gate_analysis = cls._analyze_projection(gate_w, "gate")
        logger.info("Analyzing up projection...")
        up_analysis = cls._analyze_projection(up_w, "up")
        logger.info("Analyzing down projection...")
        down_analysis = cls._analyze_projection(down_w, "down")

        # Compute similarities using down projection (matches experiment)
        logger.info("Computing expert similarities...")
        mean_sim, std_sim, sim_matrix = cls._compute_similarities(down_w)

        # Compute expert norm variance (additional signal)
        expert_norm_variance = cls._compute_expert_norm_variance(down_w)

        # Classify based on gate rank and similarity
        moe_type, confidence, training_signals = cls._classify(
            gate_rank_ratio=gate_analysis.rank_ratio,
            similarity=mean_sim,
            expert_norm_variance=expert_norm_variance,
        )

        return MoETypeAnalysis(
            model_id=model_id,
            layer_idx=layer_idx,
            num_experts=num_experts,
            moe_type=moe_type,
            architecture=architecture,
            gate=gate_analysis,
            up=up_analysis,
            down=down_analysis,
            mean_cosine_similarity=mean_sim,
            std_cosine_similarity=std_sim,
            confidence=confidence,
            training_signals=training_signals,
            similarity_matrix=sim_matrix,
        )

    @classmethod
    def _classify(
        cls,
        gate_rank_ratio: float,
        similarity: float,
        expert_norm_variance: float = 0.0,
    ) -> tuple[MoEType, float, TrainingOriginSignals]:
        """Classify MoE type from gate rank ratio and similarity.

        Args:
            gate_rank_ratio: Effective rank / max rank for gate projection
            similarity: Mean pairwise cosine similarity
            expert_norm_variance: Variance in expert weight norms

        Returns:
            Tuple of (MoEType, confidence, TrainingOriginSignals)
        """
        # Build training signals
        signals = TrainingOriginSignals(
            expert_similarity=similarity,
            rank_ratio=gate_rank_ratio,
            expert_norm_variance=expert_norm_variance,
        )

        # Compute scores
        upcycled_score = signals.upcycled_score
        pretrained_score = signals.pretrained_score

        # Classify based on scores
        if upcycled_score > 0.6 and upcycled_score > pretrained_score:
            # Upcycled: low rank (shared gate) + high similarity (clustered experts)
            confidence = min(1.0, upcycled_score)
            return MoEType.UPCYCLED, confidence, signals

        if pretrained_score > 0.6 and pretrained_score > upcycled_score:
            # Pretrained MoE: high rank (diverse gates) + low similarity (orthogonal experts)
            confidence = min(1.0, pretrained_score)
            return MoEType.PRETRAINED_MOE, confidence, signals

        # Unknown - compute confidence as how far from clear classification
        confidence = max(0.0, 1.0 - abs(upcycled_score - pretrained_score) - 0.3)
        return MoEType.UNKNOWN, confidence, signals

    @classmethod
    def _load_model(cls, model_id: str) -> nn.Module:
        """Load model using lazarus HFLoader (handles all architectures)."""
        import json

        from ...inference.loader import DType, HFLoader
        from ...models_v2.families.registry import detect_model_family, get_family_info

        result = HFLoader.download(model_id)
        model_path = result.model_path

        with open(model_path / "config.json") as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {model_id}")

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)

        return model

    @classmethod
    def _get_experts(cls, model: nn.Module, layer_idx: int):
        """Get experts object from a layer."""
        layers = _get_layers(model)
        if layer_idx >= len(layers):
            return None

        layer = layers[layer_idx]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None

        return getattr(mlp, "experts", None)

    @classmethod
    def _extract_weights(
        cls,
        experts,
        architecture: MoEArchitecture,
    ) -> tuple[mx.array, mx.array, mx.array, int]:
        """Extract gate, up, down weights from experts.

        Returns:
            (gate_weights, up_weights, down_weights, num_experts)
            Each weight array has shape (num_experts, out_dim, in_dim)
        """

        # GPT-OSS: batched quantized experts
        if architecture == MoEArchitecture.GPT_OSS and hasattr(experts, "gate_up_proj_blocks"):
            return cls._extract_batched_weights(experts)

        # Standard: list of expert modules
        if isinstance(experts, list):
            return cls._extract_list_weights(experts)

        raise ValueError(f"Unknown expert structure for architecture {architecture}")

    @classmethod
    def _extract_batched_weights(cls, experts) -> tuple[mx.array, mx.array, mx.array, int]:
        """Extract weights from GPT-OSS batched/quantized experts (MXFP4).

        Note: Biases are NOT included in the returned weights. Biases should be
        applied separately after matrix multiplication during inference.
        """
        import mlx.core as mx

        num_experts = experts.num_experts

        # Dequantize gate_up (interleaved gate and up projections) using MXFP4 mode
        # Note: We do NOT add bias here - bias should be applied to output, not weights
        gate_up = mx.dequantize(
            experts.gate_up_proj_blocks,
            experts.gate_up_proj_scales,
            biases=None,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
        mx.eval(gate_up)
        gate_up = gate_up.astype(mx.float32)

        # Dequantize down using MXFP4 mode
        # Note: We do NOT add bias here - bias should be applied to output, not weights
        down = mx.dequantize(
            experts.down_proj_blocks,
            experts.down_proj_scales,
            biases=None,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
        mx.eval(down)
        down = down.astype(mx.float32)

        # Split interleaved gate and up
        gate = gate_up[:, 0::2, :]  # Even indices
        up = gate_up[:, 1::2, :]  # Odd indices

        return gate, up, down, num_experts

    @classmethod
    def _extract_biases(cls, experts) -> tuple[mx.array | None, mx.array | None, mx.array | None]:
        """Extract biases from GPT-OSS batched experts.

        Returns:
            (gate_bias, up_bias, down_bias) - each of shape (num_experts, out_dim) or None
        """
        import mlx.core as mx

        gate_bias = None
        up_bias = None
        down_bias = None

        if hasattr(experts, "gate_up_proj_bias") and experts.gate_up_proj_bias is not None:
            # gate_up_proj_bias is interleaved: (num_experts, 2 * intermediate_size)
            gate_up_bias = experts.gate_up_proj_bias.astype(mx.float32)
            gate_bias = gate_up_bias[:, 0::2]  # Even indices
            up_bias = gate_up_bias[:, 1::2]  # Odd indices

        if hasattr(experts, "down_proj_bias") and experts.down_proj_bias is not None:
            down_bias = experts.down_proj_bias.astype(mx.float32)

        return gate_bias, up_bias, down_bias

    @classmethod
    def _extract_list_weights(cls, experts: list) -> tuple[mx.array, mx.array, mx.array, int]:
        """Extract weights from list-based experts (Mixtral, OLMoE, etc.)."""
        import mlx.core as mx

        num_experts = len(experts)

        gate_weights = []
        up_weights = []
        down_weights = []

        for expert in experts:
            if hasattr(expert, "gate_proj"):
                gate_weights.append(expert.gate_proj.weight.astype(mx.float32))
            if hasattr(expert, "up_proj"):
                up_weights.append(expert.up_proj.weight.astype(mx.float32))
            if hasattr(expert, "down_proj"):
                down_weights.append(expert.down_proj.weight.astype(mx.float32))

        gate = mx.stack(gate_weights, axis=0)
        up = mx.stack(up_weights, axis=0)
        down = mx.stack(down_weights, axis=0)

        mx.eval(gate)
        mx.eval(up)
        mx.eval(down)

        return gate, up, down, num_experts

    @classmethod
    def _analyze_projection(
        cls,
        weights: mx.array,
        name: str,
    ) -> ProjectionRankAnalysis:
        """Analyze a projection's SVD structure.

        Computes the effective rank (rank needed for 95% variance) of expert
        deltas from the mean expert. Samples a subset of experts for large models.

        Args:
            weights: Shape (num_experts, out_dim, in_dim)
            name: Projection name for labeling

        Returns:
            ProjectionRankAnalysis with rank metrics
        """
        import mlx.core as mx

        num_experts, out_dim, in_dim = weights.shape
        max_rank = min(out_dim, in_dim)

        # Sample experts for large models (SVD on 2880x2880 is slow)
        if num_experts > cls.MAX_EXPERTS_TO_SAMPLE:
            # Evenly spaced sample
            step = num_experts // cls.MAX_EXPERTS_TO_SAMPLE
            expert_indices = list(range(0, num_experts, step))[: cls.MAX_EXPERTS_TO_SAMPLE]
            logger.debug(f"Sampling {len(expert_indices)}/{num_experts} experts for {name}")
        else:
            expert_indices = list(range(num_experts))

        # Compute base (mean expert) - use all experts for accurate base
        base = mx.mean(weights, axis=0)
        mx.eval(base)

        # Compute effective rank for sampled experts
        ranks = []
        for idx, i in enumerate(expert_indices):
            delta = weights[i] - base
            mx.eval(delta)
            delta_np = np.array(delta.astype(mx.float32))

            try:
                _, S, _ = np.linalg.svd(delta_np, full_matrices=False)
                rank = cls._compute_effective_rank(S)
            except np.linalg.LinAlgError:
                rank = 0

            ranks.append(rank)
            logger.debug(f"Expert {i} {name}: rank={rank}/{max_rank}")

        mean_rank = int(np.mean(ranks)) if ranks else 0

        return ProjectionRankAnalysis(
            name=name,
            shape=(out_dim, in_dim),
            max_rank=max_rank,
            effective_rank_95=mean_rank,
        )

    @classmethod
    def _compute_effective_rank(cls, S: np.ndarray) -> int:
        """Compute rank needed to capture VARIANCE_THRESHOLD of total variance.

        Args:
            S: Singular values from SVD

        Returns:
            Effective rank (number of singular values needed)
        """
        total = np.sum(S**2)
        if total == 0:
            return 0
        cumsum = np.cumsum(S**2) / total
        return int(np.searchsorted(cumsum, cls.VARIANCE_THRESHOLD) + 1)

    @classmethod
    def _compute_similarities(
        cls, weights: mx.array
    ) -> tuple[float, float, tuple[tuple[float, ...], ...]]:
        """Compute pairwise cosine similarities between experts.

        Args:
            weights: Shape (num_experts, out_dim, in_dim)

        Returns:
            (mean_similarity, std_similarity, similarity_matrix)
        """
        import mlx.core as mx

        num_experts = weights.shape[0]

        # Flatten each expert's weights
        flat = weights.reshape(num_experts, -1)
        mx.eval(flat)
        flat_np = np.array(flat.astype(mx.float32))

        # Normalize
        norms = np.linalg.norm(flat_np, axis=1, keepdims=True)
        normalized = flat_np / (norms + 1e-10)

        # Compute full similarity matrix
        sim_matrix = np.dot(normalized, normalized.T)

        # Extract upper triangle for statistics (excluding diagonal)
        sims = []
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                sims.append(float(sim_matrix[i, j]))

        if not sims:
            empty_matrix = tuple(tuple(0.0 for _ in range(num_experts)) for _ in range(num_experts))
            return 0.0, 0.0, empty_matrix

        # Convert matrix to nested tuples for Pydantic
        matrix_tuple = tuple(tuple(float(v) for v in row) for row in sim_matrix)

        return float(np.mean(sims)), float(np.std(sims)), matrix_tuple

    @classmethod
    def _compute_expert_norm_variance(cls, weights: mx.array) -> float:
        """Compute variance in expert weight norms.

        Low variance suggests upcycling (experts started from same initialization).
        High variance suggests native pretraining (experts developed independently).

        Args:
            weights: Shape (num_experts, out_dim, in_dim)

        Returns:
            Normalized variance of expert norms (0 = identical, higher = more varied)
        """
        import mlx.core as mx

        num_experts = weights.shape[0]

        # Compute norm of each expert
        flat = weights.reshape(num_experts, -1)
        mx.eval(flat)
        flat_np = np.array(flat.astype(mx.float32))

        norms = np.linalg.norm(flat_np, axis=1)
        mean_norm = np.mean(norms)

        # Coefficient of variation (normalized variance)
        if mean_norm > 0:
            return float(np.std(norms) / mean_norm)
        return 0.0
