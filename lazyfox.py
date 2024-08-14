import os
import shutil
import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from batch_generation.pretrain_batch import tokenize_and_batch
from utils.tokenizer_loader import load_tokenizer
from training.trainer import Trainer
from dataset.train_batch_dataset import TrainBatchDataset
from models.architectures.lazyfox.lazyfox_loss_function import chukloss
from models.architectures.lazyfox.simple_language_model import SimpleLanguageModel
from models.model_config import ModelConfig

def clear_checkpoint_directory(output_directory):
    """Clear the output directory."""
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

def main(regenerate_batches, prompt):
    # Settings
    input_files = ['./sample_data/lazyfox/lazyfox_train.jsonl']
    output_dir = './output/lazyfox'
    batch_output_dir = f'{output_dir}/batches'
    checkpoint_output_dir = f'{output_dir}/checkpoints'
    batchfile_prefix = 'lazyfox'
    max_sequence_length = 128  # Example max sequence length
    batch_size = 32  # Example batch size

    # Clear the output directory
    clear_checkpoint_directory(checkpoint_output_dir)

    # Check if batches exist, if not or if regenerate_batches is True, generate them
    if regenerate_batches or not os.path.exists(batch_output_dir) or len(os.listdir(batch_output_dir)) == 0:
        print("Generating batches...")
        tokenize_and_batch(
            input_files=input_files,
            tokenizer_name='lazyfox_tokenizer',
            output_directory=batch_output_dir,
            file_prefix=batchfile_prefix,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            print_summaries=True
        )
    else:
        print("Batch files found. Skipping batch generation...")

    # Load tokenizer and define vocabulary size
    tokenizer_name = 'lazyfox_tokenizer'
    tokenizer = load_tokenizer(tokenizer_name)
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer.vocab)

    # Ensure bos_token_id and eos_token_id are integers
    bos_token_id = tokenizer.bos_token_id[0] if isinstance(tokenizer.bos_token_id, list) else tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id[0] if isinstance(tokenizer.eos_token_id, list) else tokenizer.eos_token_id

    # Print to verify values
    print(f'bos_token_id: {bos_token_id}, eos_token_id: {eos_token_id}')

    # Set the config for simple language model
    config_settings = {
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "hidden_act": "silu",
        "bos_token_id": int(bos_token_id),  # Begin-of-sequence token ID
        "eos_token_id": int(eos_token_id),  # End-of-sequence token ID
        "max_position_embeddings": max_sequence_length  # Maximum sequence length
    }

    # Load the model config
    model_config = ModelConfig.from_dict(config_settings)

    # Load the simple language model
    model = SimpleLanguageModel(model_config)

    # Define the optimizer
    learning_rate = 0.01
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Create value and grad function for loss
    loss_function = nn.value_and_grad(model, chukloss)

    # Load the batch data
    batch_dataset = TrainBatchDataset(batch_output_dir, batchfile_prefix)

    # Create an instance of the Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        loss_function=loss_function,
        progress_interval=10,
        checkpoint_dir=checkpoint_output_dir,
        checkpoint_freq_epochs=100,  # Set to save checkpoints every 100 epochs
        warmup_steps=0  # Adjust warmup steps if needed
    )

    # Set the number of epochs
    num_epochs = 50

    # Train the model
    print("Starting Training\n")
    trainer.train(num_epochs, batch_dataset)
    print("\n\nCompleted Training\n")

    #################
    # Test the model with the provided prompt
    #################

    # Tokenize the input prompt
    input_indices = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = mx.array([input_indices])

    # Forward pass to generate the sequence
    output = model(input_tensor)
    predicted_ids = mx.argmax(output, axis=-1).tolist()[0]  # Flatten the list

    # Post-process the predicted IDs to stop at <eos>
    if eos_token_id in predicted_ids:
        eos_index = predicted_ids.index(eos_token_id)
        predicted_ids = predicted_ids[:eos_index + 1]  # Include the eos token

    # Convert predicted IDs to tokens
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

    # Show the prediction
    print(f"Input sequence: {prompt}")
    print(f"Predicted sequence: {' '.join(predicted_tokens)}")
    print(f"Predicted token IDs: {predicted_ids}")

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="LazyFox Training Script")
    
    # arguments
    parser.add_argument("--regenerate-batches", action="store_true", help="Regenerate the batches before training.")
    parser.add_argument("--prompt", type=str, default="the quick brown", help="Prompt to test the model after training.")

    # parse
    args = parser.parse_args()

    # start
    main(args.regenerate_batches, args.prompt)
