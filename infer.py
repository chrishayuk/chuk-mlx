import argparse
import time
from utils.model_loader import load_model_tokenizer_and_checkpoint
from models.inference_utility import generate_response

def main(model_path, tokenizer_path, checkpoint_path, initial_prompt, system_prompt):
    # Load the model and tokenizer, and optionally a checkpoint
    model, tokenizer = load_model_tokenizer_and_checkpoint(model_path, checkpoint_path, tokenizer_path)

    # Initialize conversation context with the system prompt
    conversation_context = [
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
    ]

    # If an initial prompt is provided, use it; otherwise, enter interactive mode
    if initial_prompt:
        prompt_user(model, tokenizer, initial_prompt, conversation_context)
    else:
        interactive_mode(model, tokenizer, conversation_context)

def prompt_user(model, tokenizer, prompt, conversation_context):
    # setup the conversation context
    conversation_context.append(f"{prompt} [/INST]\n")
    context = ''.join(conversation_context)

    # start the timer
    start_time = time.time()

    # generate a response
    tokens = generate_response(model, context, tokenizer)

    # end the timer
    end_time = time.time()

    if tokens is not None and len(tokens) > 0:
        response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        total_time = end_time - start_time
        num_tokens = len(tokens)
        tokens_per_second = num_tokens / total_time if total_time > 0 else float('inf')
        conversation_context.append(f"<s>[INST]\n{response}\n[/INST]\n")
        
        # generation summary
        print(f"\nGenerated {num_tokens} tokens in {total_time:.2f} seconds ({tokens_per_second:.2f} tokens/second)\n")
    else:
        # no tokens
        print("No tokens generated")

def interactive_mode(model, tokenizer, conversation_context):
    while True:
        user_input = input("Enter your prompt (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting interactive mode.")
            break
        print("\n")
        prompt_user(model, tokenizer, user_input, conversation_context)

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="MLX Inference")

    # Arguments
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="The path to the local model directory or Hugging Face repo.")
    parser.add_argument("--tokenizer", default=None, help="The name or path for the tokenizer; if not specified, use the model path.")
    parser.add_argument("--checkpoint", default=None, help="The path to a checkpoint to load the model weights from.")
    parser.add_argument("--prompt", default=None, help="The initial prompt")
    parser.add_argument("--system_prompt", default="You are a helpful assistant", help="The system prompt to set the assistant's behavior")
    
    # Parse arguments
    args = parser.parse_args()

    # call main
    main(args.model, args.tokenizer, args.checkpoint, args.prompt, args.system_prompt)