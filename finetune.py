import argparse
import mlx.core as mx
import mlx.nn as nn
from trainer import Trainer
from chuk_loss_function.chuk_loss_function import chukloss
from batches.dataset.finetune_batch_dataset import FineTuneBatchDataset
from utils.training_config_loader import load_training_config
from utils.model_loader import load_model_and_tokenizer
from utils.optimizer_loader import load_optimizer

def load_configurations(config_file):
    """Load configurations from YAML file."""
    config = load_training_config(config_file)

    # get the config sections
    model_config = config['model']
    optimizer_config = config['optimizer']
    checkpoint_config = config['checkpoint']
    training_config = config['training']
    batch_config = config['batch']

    # return the config
    return model_config, optimizer_config, checkpoint_config, training_config, batch_config

def create_trainer_instance(model, tokenizer, optimizer_config, checkpoint_config, training_config):
    """Create an instance of the Trainer."""
    total_iterations = training_config['total_iterations']
    optimizer = load_optimizer(optimizer_config, total_iterations)
    loss_function = nn.value_and_grad(model, chukloss)
    lr_schedule_warmup_steps = int(optimizer_config['lr_schedule'].get('warmup_steps', 0))

    trainer = Trainer(
        model, tokenizer, optimizer, loss_function, 
        training_config['num_epochs'], 
        checkpoint_config['output_dir'], 
        checkpoint_config['frequency'], 
        lr_schedule_warmup_steps
    )
    return trainer

def load_batch_data(batch_config):
    """Load the batch data."""
    batch_dataset = FineTuneBatchDataset(batch_config['output_dir'], batch_config['file_prefix'])
    return batch_dataset

def main():
    # setup the argument parser
    parser = argparse.ArgumentParser(description="Fine-tune a model using specified configuration.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--iterations', type=int, help='Override the total number of iterations.')
    parser.add_argument('--epochs', type=int, help='Override the number of epochs.')

    # parse arguments
    args = parser.parse_args()

    # Load configurations
    model_config, optimizer_config, checkpoint_config, training_config, batch_config = load_configurations(args.config)

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config['name'])

    # Override iterations and epochs if provided
    if args.iterations:
        training_config['total_iterations'] = args.iterations
    if args.epochs:
        training_config['num_epochs'] = args.epochs

    # Create an instance of the Trainer
    trainer = create_trainer_instance(model, tokenizer, optimizer_config, checkpoint_config, training_config)

    # Load batch data
    batch_dataset = load_batch_data(batch_config)

    # Train the model
    trainer.train(training_config['num_epochs'], batch_dataset, training_config['total_iterations'])

if __name__ == "__main__":
    main()
