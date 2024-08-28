import argparse
import os
import shutil
import mlx.core as mx
import mlx.nn as nn
import logging
from core.utils.tokenizer_loader import load_tokenizer
from training.trainer import Trainer
from core.models.chuk_loss_function import chukloss
from core.models.loss_function_loader import load_loss_function
from core.dataset.train_batch_dataset import TrainBatchDataset
from core.utils.training_config_loader import load_training_config
from core.models.model_loader import load_model
from core.utils.optimizer_loader import load_optimizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_configurations(config_file: str):
    """Load and validate configurations from YAML file."""
    config = load_training_config(config_file)
    required_sections = ['model', 'optimizer', 'checkpoint', 'training', 'batch']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Configuration file missing '{section}' section")
    
    # Set default loss function if not specified
    if 'loss_function' not in config['training']:
        logger.info("No loss function specified in configuration, using default 'chukloss'.")
        config['training']['loss_function'] = 'chukloss'

    # Set default tokenizer if not specified
    if 'tokenizer' not in config['model']:
        logger.info(f"No tokenizer specified in model configuration, using model name '{config['model']['name']}' as tokenizer.")
        config['model']['tokenizer'] = config['model']['name']

    # Set default load_initial_weights to True if not specified
    if 'load_initial_weights' not in config['model']:
        config['model']['load_initial_weights'] = True
    else:
        config['model']['load_initial_weights'] = config['model'].get('load_initial_weights', True)
    
    return config['model'], config['optimizer'], config['checkpoint'], config['training'], config['batch']

def create_trainer_instance(model, tokenizer, optimizer_config, checkpoint_config, training_config):
    """Create an instance of the Trainer."""
    total_iterations = training_config.get('total_iterations', 1000)
    optimizer = load_optimizer(optimizer_config, total_iterations)

    # Load the specified loss function
    if training_config['loss_function'] == 'chukloss':
        loss_function = nn.value_and_grad(model, chukloss)
    else:
        loss_function = nn.value_and_grad(model, load_loss_function(training_config['loss_function']))

    lr_schedule_warmup_steps = int(optimizer_config['lr_schedule'].get('warmup_steps', 0))

    trainer = Trainer(
        model, tokenizer, optimizer, loss_function, 
        progress_interval=training_config.get('progress_interval', 1),
        checkpoint_dir=checkpoint_config['output_dir'], 
        checkpoint_freq_epochs=checkpoint_config.get('frequency_epochs', None),
        checkpoint_freq_iterations=checkpoint_config.get('frequency_iterations', None),
        warmup_steps=lr_schedule_warmup_steps
    )
    return trainer

def load_batch_data(batch_config):
    """Load the batch data."""
    return TrainBatchDataset(batch_config['output_dir'], batch_config['file_prefix'], pre_cache_size=batch_config['pre_cache_size'])

def clear_output_checkpoints(checkpoint_dir):
    """Clear the output checkpoints directory."""
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)

def main():
    parser = argparse.ArgumentParser(description="Train a model using specified configuration.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--iterations', type=int, help='Override the total number of iterations.')
    parser.add_argument('--epochs', type=int, help='Override the number of epochs.')
    args = parser.parse_args()

    try:
        model_config, optimizer_config, checkpoint_config, training_config, batch_config = load_configurations(args.config)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    # Clear the output checkpoints directory
    clear_output_checkpoints(checkpoint_config['output_dir'])

    # load the tokenizer
    tokenizer = load_tokenizer(model_config['tokenizer'])

    #try:
    # load the model
    model = load_model(model_config['name'], model_config['load_initial_weights'] )
    #except Exception as e:
    #    return
    #    logger.error(f"Error loading model and tokenizer: {e}")

    if args.iterations:
        training_config['total_iterations'] = args.iterations

    if args.epochs:
        training_config['num_epochs'] = args.epochs

    #try:
    trainer = create_trainer_instance(model, tokenizer, optimizer_config, checkpoint_config, training_config)
    # except Exception as e:
    #     logger.error(f"Error creating trainer instance: {e}")
    #     return

    try:
        batch_dataset = load_batch_data(batch_config)
    except Exception as e:
        logger.error(f"Error loading batch data: {e}")
        return

    #try:
    trainer.train(training_config['num_epochs'], batch_dataset, training_config['total_iterations'])
    #except Exception as e:
    #    logger.error(f"Error during training: {e}")

if __name__ == "__main__":
    main()
