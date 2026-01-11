"""Calculate tokenizer-dataset fit score command handler."""

import logging

from .._types import AnalyzeFitScoreConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def analyze_fit_score(config: AnalyzeFitScoreConfig) -> None:
    """Calculate tokenizer-dataset fit score.

    Args:
        config: Fit score configuration.
    """
    from .....data.tokenizers.analyze import calculate_fit_score
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Calculating fit score on {len(texts)} texts...")
    score = calculate_fit_score(texts, tokenizer)

    print("\n=== Fit Score Report ===")
    print(f"Overall Score:     {score.score:.2f}/100")
    print(f"Grade:             {score.grade}")

    if score.recommendations:
        print("\nRecommendations:")
        for rec in score.recommendations:
            print(f"  - {rec}")

    if score.details:
        print("\nDetails:")
        for key, val in score.details.items():
            print(f"  {key}: {val}")
