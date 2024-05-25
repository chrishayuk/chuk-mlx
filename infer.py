import argparse
import time
from utils.model_loader import load_model_tokenizer_and_checkpoint
from models.inference_utility import generate_response

def main(model_path, tokenizer_path, checkpoint_path, initial_prompt, system_prompt, chat_mode):
    # Load the model and tokenizer, and optionally a checkpoint
    model, tokenizer = load_model_tokenizer_and_checkpoint(model_path, checkpoint_path, tokenizer_path)

    if chat_mode:
        # Initialize conversation context with the system prompt
        conversation_context = [
            {"role": "system", "content": system_prompt}
        ]

        # If an initial prompt is provided, use it; otherwise, enter interactive mode
        if initial_prompt:
            prompt_user_chat(model, tokenizer, initial_prompt, conversation_context)
        else:
            interactive_mode_chat(model, tokenizer, conversation_context)
    else:
        # If an initial prompt is provided, use it; otherwise, enter interactive mode
        if initial_prompt:
            prompt_user(model, tokenizer, initial_prompt)
        else:
            interactive_mode(model, tokenizer)

def generate_and_time_response(model, tokenizer, context):
    # Start the timer
    start_time = time.time()

    # Generate a response
    tokens = generate_response(model, context, tokenizer)

    # End the timer
    end_time = time.time()

    return tokens, start_time, end_time

def print_generation_summary(tokens, start_time, end_time):
    # Ensure we have tokens
    if tokens is not None and len(tokens) > 0:
        # Calculate the timings
        total_time = end_time - start_time
        num_tokens = len(tokens)
        tokens_per_second = num_tokens / total_time if total_time > 0 else float('inf')

        # Generation summary
        print("\n")
        print("=" * 50)
        print(f"Generated {num_tokens} tokens in {total_time:.2f} seconds ({tokens_per_second:.2f} tokens/second)")
        print("=" * 50)
    else:
        # No tokens
        print("No tokens generated")

def prompt_user(model, tokenizer, prompt):
    tokens, start_time, end_time = generate_and_time_response(model, tokenizer, prompt)
    print_generation_summary(tokens, start_time, end_time)

def interactive_mode(model, tokenizer):
    while True:
        user_input = input("Enter your prompt (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting interactive mode.")
            break
        print("\n")
        prompt_user(model, tokenizer, user_input)

def prompt_user_chat(model, tokenizer, prompt, conversation_context):
    # Setup the conversation context
    conversation_context.append({"role": "user", "content": prompt})

    # Apply the chat template to the conversation context
    tokenized_chat = tokenizer.apply_chat_template(conversation_context, tokenize=True, return_tensors="pt")
    context = tokenizer.decode(tokenized_chat[0])

    tokens, start_time, end_time = generate_and_time_response(model, tokenizer, context)

    # Ensure we have tokens
    if tokens is not None and len(tokens) > 0:
        # Get and decode the response
        response = tokenizer.decode(tokens, skip_special_tokens=True).strip()

        # Add the response to the context
        conversation_context.append({"role": "assistant", "content": response})

    print_generation_summary(tokens, start_time, end_time)

def interactive_mode_chat(model, tokenizer, conversation_context):
    while True:
        user_input = input("Enter your prompt (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting interactive mode.")
            break
        print("\n")
        prompt_user_chat(model, tokenizer, user_input, conversation_context)

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="MLX Inference")

    # Arguments
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="The path to the local model directory or Hugging Face repo.")
    parser.add_argument("--tokenizer", default=None, help="The name or path for the tokenizer; if not specified, use the model path.")
    parser.add_argument("--checkpoint", default=None, help="The path to a checkpoint to load the model weights from.")
    parser.add_argument("--prompt", default=None, help="The initial prompt")
    parser.add_argument("--system_prompt", default="You are a helpful assistant", help="The system prompt to set the assistant's behavior")
    parser.add_argument("--chat", action='store_true', help="Use chat mode with the Hugging Face chat template")
    parser.add_argument("--text", action='store_true', help="Use text mode for direct text generation")

    # Parse arguments
    args = parser.parse_args()

    # Determine the mode based on the provided arguments
    if args.chat:
        chat_mode = True
    elif args.text:
        chat_mode = False
    else:
        # Default to chat mode for interactive mode, text mode for non-interactive mode
        chat_mode = args.prompt is None

    # Call main
    main(args.model, args.tokenizer, args.checkpoint, args.prompt, args.system_prompt, chat_mode)
