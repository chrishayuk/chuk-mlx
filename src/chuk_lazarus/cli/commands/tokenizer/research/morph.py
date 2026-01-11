"""Token morphing research command handler."""

import json
import logging

import numpy as np

from .._types import ResearchMorphConfig

logger = logging.getLogger(__name__)


def research_morph(config: ResearchMorphConfig) -> None:
    """Morph between token embeddings.

    Args:
        config: Morph configuration.
    """
    from .....data.tokenizers.research import (
        MorphConfig as InternalMorphConfig,
    )
    from .....data.tokenizers.research import (
        MorphMethod,
        compute_path_length,
        compute_straightness,
        morph_token,
    )

    # Load embeddings
    with open(config.file) as f:
        data = json.load(f)

    embeddings = np.array(data["embeddings"], dtype=np.float32)
    token_strs = data.get("token_strs", [f"token_{i}" for i in range(len(embeddings))])

    if config.source >= len(embeddings) or config.target >= len(embeddings):
        logger.error(f"Source/target index out of range (max: {len(embeddings) - 1})")
        return

    method = MorphMethod(config.method.value)
    internal_config = InternalMorphConfig(
        method=method,
        num_steps=config.steps,
        include_endpoints=True,
        normalize_output=config.normalize,
    )

    source_emb = embeddings[config.source]
    target_emb = embeddings[config.target]

    result = morph_token(
        source_emb,
        target_emb,
        token_strs[config.source],
        token_strs[config.target],
        internal_config,
    )

    print("\n=== Token Morphing ===")
    print(f"Source:      {result.source_token}")
    print(f"Target:      {result.target_token}")
    print(f"Method:      {result.method.value}")
    print(f"Steps:       {result.num_steps}")
    print(f"Path length: {compute_path_length(result):.4f}")
    print(f"Straightness: {compute_straightness(result):.4f}")

    trajectory = result.get_embeddings_array()
    print("\nTrajectory norms:")
    for i, alpha in enumerate(result.alphas):
        norm = np.linalg.norm(trajectory[i])
        print(f"  alpha={alpha:.2f}: norm={norm:.4f}")

    if config.output:
        output_data = {
            "source": result.source_token,
            "target": result.target_token,
            "method": result.method.value,
            "alphas": result.alphas,
            "embeddings": result.embeddings,
        }
        with open(config.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved trajectory to: {config.output}")
