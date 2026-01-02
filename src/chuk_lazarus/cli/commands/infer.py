"""Inference command handlers for chuk-lazarus CLI."""

import logging

logger = logging.getLogger(__name__)


def run_inference(args):
    """Run inference on a model."""
    from ...models import load_model

    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model)

    if args.adapter:
        logger.info(f"Loading adapter: {args.adapter}")
        model.load_adapter(args.adapter)

    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        prompts = []
        print("Enter prompts (Ctrl+D to finish):")
        try:
            while True:
                prompt = input("> ")
                if prompt:
                    prompts.append(prompt)
        except EOFError:
            pass

    for prompt in prompts:
        response = model.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
