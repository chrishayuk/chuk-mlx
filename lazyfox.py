import os
from pathlib import Path
import shutil
import argparse
from core.models.architectures.lazyfox.lazyfox_loss_function import chukloss
from core.models.architectures.lazyfox.lazyfox_model import CustomModel
from core.utils.model_adapter import ModelAdapter
import mlx.optimizers as optim
from batch_generation.pretrain_batch import tokenize_and_batch
from core.utils.tokenizer_loader import load_tokenizer
from training.trainer import Trainer
from dataset.train_batch_dataset import TrainBatchDataset
from core.models.model_config import ModelConfig

def clear_checkpoint_directory(output_directory):
    """Clear the output directory."""
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

def main(regenerate_batches, prompt, model_name, tokenizer_name, framework='mlx'):
    # Settings
    input_files = [f'./sample_data/{tokenizer_name}/{tokenizer_name}_train.jsonl']
    output_dir = f'./output/{model_name}'
    batch_output_dir = f'{output_dir}/batches'
    checkpoint_output_dir = f'{output_dir}/checkpoints'
    batchfile_prefix = model_name
    max_sequence_length = 15  # Example max sequence length
    batch_size = 2   # Example batch size

    # Clear the output directory
    clear_checkpoint_directory(checkpoint_output_dir)

    # Load tokenizer and define vocabulary size
    tokenizer = load_tokenizer(tokenizer_name)

    # Check if batches exist, if not or if regenerate_batches is True, generate them
    if regenerate_batches or not os.path.exists(batch_output_dir) or len(os.listdir(batch_output_dir)) == 0:
        print("Generating batches...")
        tokenize_and_batch(
            input_files=input_files,
            tokenizer=tokenizer,
            output_directory=batch_output_dir,
            file_prefix=batchfile_prefix,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            print_summaries=True
        )
    else:
        print("Batch files found. Skipping batch generation...")

    # Load the model config
    config_path = Path(f"./models/architectures/{model_name}/config.json")
    model_config = ModelConfig.load(config_path)

    # Initialize the ModelAdapter with the specified framework
    model_adapter = ModelAdapter(framework=framework)

    # Load the appropriate model
    model = CustomModel(model_config)
    model_adapter.model = model

    # Define the optimizer
    learning_rate = 0.01
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Create value and grad function for loss
    loss_function = model_adapter.create_value_and_grad_fn(chukloss)

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
    input_tensor = model_adapter.to_tensor([input_indices])

    # Forward pass to generate the sequence
    output = model_adapter.forward(input_tensor)
    predicted_ids = model_adapter.argmax(output, axis=-1)

    # If predicted_ids is a list of lists, flatten it
    if isinstance(predicted_ids[0], list):
        predicted_ids = [item for sublist in predicted_ids for item in sublist]

    # Post-process the predicted IDs to stop at <eos>
    if model_config.eos_token_id in predicted_ids:
        eos_index = predicted_ids.index(model_config.eos_token_id)
        predicted_ids = predicted_ids[:eos_index + 1]  # Include the eos token

    # Convert predicted IDs to tokens
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

    # Show the prediction
    print(f"Input sequence: {prompt}")
    print(f"Predicted sequence: {' '.join(predicted_tokens)}")
    print(f"Predicted token IDs: {predicted_ids}")

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="Model Training Script")
    
    # arguments
    parser.add_argument("--regenerate-batches", action="store_true", help="Regenerate the batches before training.")
    parser.add_argument("--prompt", type=str, default="the quick brown", help="Prompt to test the model after training.")
    parser.add_argument("--model-name", type=str, default="lazyfox", help="Name of the model to load (e.g., lazyfox, math).")
    parser.add_argument("--tokenizer-name", type=str, default="lazyfox", help="Name of the tokenizer to load and use for training data.")
    parser.add_argument("--framework", type=str, default="mlx", choices=["mlx", "torch"], help="Framework to use for training and inference.")

    # parse
    args = parser.parse_args()

    # start
    main(args.regenerate_batches, args.prompt, args.model_name, args.tokenizer_name, args.framework)
