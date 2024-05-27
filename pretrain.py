import argparse
import mlx.core as mx
import mlx.nn as nn
from models.loss_function_loader import load_loss_function
from training.trainer import Trainer
# from models.chuk_loss_function import chukloss
#from models.architectures.lazyfox.lazyfox_loss_function import chukloss
from dataset.pretrain.pretrain_batch_dataset import PreTrainBatchDataset
from utils.training_config_loader import load_training_config
from models.model_loader import load_model_and_tokenizer
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
    
    if 'loss_function' not in config['training']:
        raise ValueError("Training configuration missing 'loss_function' entry")
       
    # return the config sections
    return config['model'], config['optimizer'], config['checkpoint'], config['training'], config['batch']

def create_trainer_instance(model, tokenizer, optimizer_config, checkpoint_config, training_config):
    """Create an instance of the Trainer."""
    
    # get the total iterations
    total_iterations = training_config['total_iterations']

    # load the optimizers
    optimizer = load_optimizer(optimizer_config, total_iterations)

    # set the loss function
    chukloss = load_loss_function(training_config['loss_function'])
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
    return PreTrainBatchDataset(batch_config['output_dir'], batch_config['file_prefix'])

def main():
    # setup the argument parser
    parser = argparse.ArgumentParser(description="Pre-train a model using specified configuration.")

    # get the arguments
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--iterations', type=int, help='Override the total number of iterations.')
    parser.add_argument('--epochs', type=int, help='Override the number of epochs.')

    # parse
    args = parser.parse_args()

    try:
        # load the configurations
        model_config, optimizer_config, checkpoint_config, training_config, batch_config = load_configurations(args.config)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    try:
        # load the model and the tokenizer
        model, tokenizer = load_model_and_tokenizer(model_config['name'], load_weights=False)
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        return

    # override the number of iterations if passed
    if args.iterations:
        training_config['total_iterations'] = args.iterations

    # override the number of epochs if passed
    if args.epochs:
        training_config['num_epochs'] = args.epochs

    try:
        # create the trainer instance
        trainer = create_trainer_instance(model, tokenizer, optimizer_config, checkpoint_config, training_config)
    except Exception as e:
        logger.error(f"Error creating trainer instance: {e}")
        return

    try:
        # load the batch data
        batch_dataset = load_batch_data(batch_config)
    except Exception as e:
        logger.error(f"Error loading batch data: {e}")
        return

    # kick off training
    trainer.train(training_config['num_epochs'], batch_dataset, training_config['total_iterations'])

if __name__ == "__main__":
    # let's go
    main()
