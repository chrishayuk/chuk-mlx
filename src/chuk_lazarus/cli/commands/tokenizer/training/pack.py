"""Training sequence packing command handler."""

import json
import logging

from .._types import PackResult, TrainingPackConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def training_pack(config: TrainingPackConfig) -> PackResult:
    """Pack sequences for efficient training.

    Args:
        config: Pack configuration.

    Returns:
        Pack result with packing statistics.
    """
    from .....data.tokenizers.training import PackingConfig, pack_sequences
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return PackResult(
            input_sequences=0,
            packed_sequences=0,
            packing_ratio=0,
            efficiency=0,
        )

    packing_config = PackingConfig(
        max_seq_length=config.max_length,
        padding_token_id=tokenizer.pad_token_id or 0,
        separator_token_id=tokenizer.eos_token_id,
    )

    logger.info(f"Packing {len(texts)} sequences to max length {config.max_length}...")
    packed = pack_sequences(texts, tokenizer, packing_config)

    total_tokens = sum(len(p.token_ids) for p in packed)
    efficiency = total_tokens / (len(packed) * config.max_length) if packed else 0
    packing_ratio = len(texts) / len(packed) if packed else 0

    print("\n=== Packing Results ===")
    print(f"Input sequences:   {len(texts)}")
    print(f"Packed sequences:  {len(packed)}")
    print(f"Packing ratio:     {packing_ratio:.2f}x" if packed else "N/A")
    print(f"Efficiency:        {efficiency:.2%}")

    output_path = None
    if config.output:
        with open(config.output, "w") as f:
            for p in packed:
                f.write(
                    json.dumps({"token_ids": p.token_ids, "boundaries": p.sequence_boundaries})
                    + "\n"
                )
        output_path = config.output
        print(f"\nSaved to: {config.output}")

    return PackResult(
        input_sequences=len(texts),
        packed_sequences=len(packed),
        packing_ratio=packing_ratio,
        efficiency=efficiency,
        output_path=output_path,
    )
