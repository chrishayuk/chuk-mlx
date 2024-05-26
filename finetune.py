import argparse
import mlx.core as mx
import mlx.nn as nn
from trainer import Trainer
from chuk_loss_function.chuk_loss_function import chukloss
from batches.dataset.finetune_batch_dataset import FineTuneBatchDataset
from utils.training_config_loader import load_training_config
from utils.model_loader import load_model_and_tokenizer
from utils.optimizer_loader import load_optimizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_configurations(config_file: str):
    """Load configurations from YAML file."""
    config = load_training_config(config_file)
    required_sections = ['model', 'optimizer', 'checkpoint', 'training', 'batch']
    
    # Validate configuration
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Configuration file missing '{section}' section")
        
    # return the config sections
    return config['model'], config['optimizer'], config['checkpoint'], config['training'], config['batch']

def create_trainer_instance(model, tokenizer, optimizer_config, checkpoint_config, training_config):
    """Create an instance of the Trainer."""

    # get the total iterations
    total_iterations = training_config['total_iterations']

    # load the optimizers
    optimizer = load_optimizer(optimizer_config, total_iterations)

    # set the loss function
    loss_function = nn.value_and_grad(model, chukloss)

    # set the warmup steps
    lr_schedule_warmup_steps = int(optimizer_config['lr_schedule'].get('warmup_steps', 0))

    # initalize the trainer
    trainer = Trainer(
        model, tokenizer, optimizer, loss_function, 
        training_config['num_epochs'], 
        checkpoint_config['output_dir'], 
        checkpoint_config['frequency'], 
        lr_schedule_warmup_steps
    )

    # return the trainer instance
    return trainer

def load_batch_data(batch_config):
    """Load the batch data."""
    return FineTuneBatchDataset(batch_config['output_dir'], batch_config['file_prefix'])

def main():
    # setup the arguent parser
    parser = argparse.ArgumentParser(description="Fine-tune a model using specified configuration.")

    # get the arguments
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--iterations', type=int, help='Override the total number of iterations.')
    parser.add_argument('--epochs', type=int, help='Override the number of epochs.')

    # paeae
    args = parser.parse_args()

    try:
        # load the configurations
        model_config, optimizer_config, checkpoint_config, training_config, batch_config = load_configurations(args.config)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # load the model and the tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config['name'])

     # override the number of iterations if passed
    if args.iterations:
        training_config['total_iterations'] = args.iterations

    # override the number of epochs if passed
    if args.epochs:
        training_config['num_epochs'] = args.epochs

    # create the trainer instance
    trainer = create_trainer_instance(model, tokenizer, optimizer_config, checkpoint_config, training_config)

    # load the batch data
    batch_dataset = load_batch_data(batch_config)

    try:
        # kick off training
        trainer.train(training_config['num_epochs'], batch_dataset, training_config['total_iterations'])
    except Exception as e:
        logger.error(f"Training error: {e}")

if __name__ == "__main__":
    # let's go
    main()
