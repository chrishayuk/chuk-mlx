"""Training throughput profiling command handler."""

import logging

from .._types import TrainingThroughputConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def training_throughput(config: TrainingThroughputConfig) -> None:
    """Profile tokenization throughput.

    Args:
        config: Throughput configuration.
    """
    from .....data.tokenizers.training import ThroughputProfiler
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Profiling throughput on {len(texts)} texts...")
    profiler = ThroughputProfiler(tokenizer)
    metrics = profiler.profile(
        texts, batch_size=config.batch_size, num_iterations=config.iterations
    )

    print("\n=== Throughput Profile ===")
    print(f"Tokens/second:     {metrics.tokens_per_second:,.0f}")
    print(f"Texts/second:      {metrics.texts_per_second:,.0f}")
    print(f"Avg batch time:    {metrics.avg_batch_time_ms:.2f} ms")
    print(f"Total tokens:      {metrics.total_tokens:,}")
    print(f"Total time:        {metrics.total_time_seconds:.2f} s")
