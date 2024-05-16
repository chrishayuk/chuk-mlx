import argparse
import time
import mlx.core as mx
import models
from models.inference_utility import generate_response
import models.llama
import models.llama.model
from models.load_weights import load_model_weights
from models.model_config import ModelConfig
from utils.huggingface_utils import load_from_hub
from utils.tokenizer_loader import load_tokenizer

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="MLX Inference")

    # Arguments
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="The path to the local model directory or Hugging Face repo.")
    parser.add_argument("--tokenizer", default=None, help="The name or path for the tokenizer; if not specified, use the model path.")
    parser.add_argument("--prompt", default="Who is Ada Lovelace?", help="The prompt")
    
    # Parse arguments
    args = parser.parse_args()

    # Load the model from Hugging Face
    print(f"Loading Model: {args.model}")
    model_path = load_from_hub(args.model)

    # Load model config
    print("Loading Model Config")
    model_config = ModelConfig.load(model_path)

    # Load model weights
    print("Loading Model Weights")
    weights = load_model_weights(model_path)

    # Create the model instance
    model = models.llama.model.Model(model_config)
    print(f"Model Loaded: {args.model}")

    # Load model weights
    print("Loading weights")
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    print("Weights loaded")

    # Load the tokenizer
    print("Loading Tokenizer")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = load_tokenizer(tokenizer_path)

    # Generate response for the prompt
    print("Prompting")
    start_time = time.time()
    tokens = generate_response(model, args.prompt, tokenizer)
    end_time = time.time()

    if tokens is not None:
        total_time = end_time - start_time
        num_tokens = len(tokens)
        tokens_per_second = num_tokens / total_time if total_time > 0 else float('inf')
        print(f"Generated {num_tokens} tokens in {total_time:.2f} seconds ({tokens_per_second:.2f} tokens/second)")
    else:
        print("No tokens generated")