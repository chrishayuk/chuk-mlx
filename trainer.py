import time
from tqdm import tqdm
import logging
import os
import mlx.core as mx

from batches.sequence_utility import SequenceUtility
from utils.tokenizer_loader import load_tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    @staticmethod
    def pad_sequences(sequences, max_length, padding_value=0):
        padded_sequences = []
        for seq in sequences:
            seq_length = seq.shape[0]
            if seq_length < max_length:
                # Create a list of padding values and concatenate it with the sequence
                padding = [padding_value] * (max_length - seq_length)
                padded_seq = mx.array(seq.tolist() + padding)
            else:
                padded_seq = seq[:max_length]  # Truncate if necessary
            padded_sequences.append(padded_seq)
        return mx.stack(padded_sequences)

    def __init__(self, model, tokenizer, optimizer, loss_function, progress_interval=1, checkpoint_dir='checkpoints', checkpoint_freq=None, warmup_steps=0):
        # initialize
        self.model = model

        # set the tokenizer
        self.tokenizer = tokenizer

        # set the other values
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.progress_interval = progress_interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.warmup_steps = warmup_steps

        # if we've set to checkpoint, create the directory
        if self.checkpoint_freq is not None and not os.path.exists(self.checkpoint_dir):
            # make the directory
            os.makedirs(self.checkpoint_dir)

    def get_current_learning_rate(self):
        # get the current learning rate from the optimizer
        lr = self.optimizer.learning_rate

        # return it
        return float(lr) if isinstance(lr, (int, float)) else lr.item()

    def save_checkpoint(self, identifier):
        # get the checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{identifier}.npz')

        # save the weights
        self.model.save_weights(checkpoint_path)

        # log the save
        logger.info(f'Saved checkpoint: {checkpoint_path}')

    def train(self, num_epochs, batch_dataset, num_iterations=None):
        # initialize
        total_batch_time = 0
        total_epoch_time = 0
        iteration_count = 0
        total_tokens = 0
        total_theoretical_tokens = 0
        checkpoint_interval = None

        # set the frequency of checkpointing
        if self.checkpoint_freq is not None:
            checkpoint_interval = max(1, (num_epochs * 100) // self.checkpoint_freq)

        # perform the number of epochs
        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(epoch, num_epochs, batch_dataset, num_iterations, iteration_count)

            # update metrics and counters
            iteration_count = epoch_metrics["iteration_count"]
            total_tokens += epoch_metrics["epoch_tokens"]
            total_theoretical_tokens += epoch_metrics["epoch_theoretical_tokens"]
            total_batch_time += epoch_metrics["total_batch_time"]
            total_epoch_time += epoch_metrics["epoch_time"]

            # check if we've exceeded the number of iterations to train for
            if num_iterations is not None and iteration_count >= num_iterations:
                break

        # calculate the tokens per second for the total training
        actual_tokens_per_second = total_tokens / total_epoch_time
        theoretical_tokens_per_second = total_theoretical_tokens / total_epoch_time

        # log out the total training time etc
        logger.info(f"Total training time: {total_epoch_time:.3f}s")
        logger.info(f"Total iterations: {iteration_count}")
        logger.info(f"Average batch time: {total_batch_time / iteration_count:.3f}s")
        logger.info(f"Tokens per second: {actual_tokens_per_second:.2f} (Actual) / {theoretical_tokens_per_second:.2f} (Theoretical)")

    def train_epoch(self, epoch, num_epochs, batch_dataset, num_iterations, iteration_count):
        # start the time for the epoch
        epoch_start_time = time.time()

        # reset the counters for the epoch
        batch_times = []
        epoch_loss = 0
        epoch_tokens = 0
        epoch_theoretical_tokens = 0
        batch_count = 0
        batch_progress = tqdm(desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

        # loop through the batches
        for batch_index, batch in enumerate(batch_dataset, start=1):
            # check if we've exceeded iterations
            if num_iterations is not None and iteration_count >= num_iterations:
                break

            batch_metrics = self.train_batch(batch, batch_index, iteration_count)

            # update metrics and counters
            epoch_loss += batch_metrics["loss"]
            epoch_tokens += batch_metrics["ntoks"]
            epoch_theoretical_tokens += batch_metrics["expected_tokens"]
            batch_times.append(batch_metrics["batch_time"])
            batch_count += 1
            iteration_count += 1

            # update progress bar
            self.update_progress_bar(batch_progress, batch_index, batch_metrics)

        # calculate epoch metrics
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        average_batch_time = sum(batch_times) / len(batch_times)
        actual_tokens_per_second = epoch_tokens / sum(batch_times)
        theoretical_tokens_per_second = epoch_theoretical_tokens / sum(batch_times)

        # log epoch metrics
        self.log_epoch_metrics(epoch, num_epochs, batch_count, epoch_loss, epoch_tokens, average_batch_time, actual_tokens_per_second, theoretical_tokens_per_second)

        # checkpoint if necessary
        self.checkpoint_if_necessary(epoch)

        return {
            "iteration_count": iteration_count,
            "epoch_tokens": epoch_tokens,
            "epoch_theoretical_tokens": epoch_theoretical_tokens,
            "total_batch_time": sum(batch_times),
            "epoch_time": epoch_time
        }

    import mlx.core as mx

    def train_batch(self, batch, batch_index, iteration_count):
        concatenated_tensor, lengths = batch

        # Check if sep_token_id is valid
        if self.tokenizer.sep_token_id is None:
            logger.error("Separator token ID (sep_token_id) is not set in the tokenizer.")
            raise ValueError("Separator token ID (sep_token_id) is not set in the tokenizer.")

        # Create empty lists to store the input and target tensors
        input_tensors = []
        target_tensors = []

        # Iterate over the sequences in the batch
        for i in range(concatenated_tensor.shape[0]):
            seq_length = int(lengths[i].item())  # Convert scalar tensor to integer
            seq_tensor = concatenated_tensor[i, :seq_length]

            # Find the first occurrence of the separator token
            sep_index = -1
            for idx in range(seq_length):
                if int(seq_tensor[idx].item()) == self.tokenizer.sep_token_id:
                    sep_index = idx
                    break

            if sep_index == -1:
                logger.warning(f"No separator found in sequence {i} of batch {batch_index}. Skipping this sequence.")
                continue

            # Slice the concatenated tensor to obtain the input and target tensors
            input_tensor = seq_tensor[:sep_index + 1]  # Include the separator token in the input
            target_tensor = seq_tensor[sep_index + 1:seq_length]

            # Pad the target tensor to match the length of the input tensor
            if target_tensor.shape[0] < input_tensor.shape[0]:
                padding_length = input_tensor.shape[0] - target_tensor.shape[0]
                padding = [self.tokenizer.pad_token_id] * padding_length
                target_tensor = mx.array(target_tensor.tolist() + padding)

            # Append the input and target tensors to the respective lists
            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)

        if not input_tensors or not target_tensors:
            logger.error(f"No valid sequences in batch {batch_index}. Skipping this batch.")
            return {
                "loss": 0,
                "ntoks": 0,
                "expected_tokens": 0,
                "batch_time": 0,
                "lr_before_update": self.optimizer.learning_rate
            }

        # Determine the maximum length for padding
        max_input_length = max(tensor.shape[0] for tensor in input_tensors)
        max_target_length = max(tensor.shape[0] for tensor in target_tensors)

        # Pad the sequences to have the same length
        input_tensor = self.pad_sequences(input_tensors, max_input_length, self.tokenizer.pad_token_id)
        target_tensor = self.pad_sequences(target_tensors, max_input_length, self.tokenizer.pad_token_id)

        # Print the shapes of the input and target tensors after padding
        #print(f"Batch {batch_index} - Padded Input shape: {input_tensor.shape}, Padded Target shape: {target_tensor.shape}")

        # Work out the expected number of tokens
        expected_tokens = input_tensor.shape[0] * input_tensor.shape[1]

        # Start the batch timer
        batch_start_time = time.time()

        try:
            # Get the current learning rate
            if iteration_count < self.warmup_steps:
                # Warmup phase
                warmup_factor = (iteration_count + 1) / self.warmup_steps
                current_lr = self.optimizer.learning_rate * warmup_factor
                self.optimizer.learning_rate = current_lr
            else:
                # Post-warmup phase
                current_lr = self.optimizer.learning_rate

            # Get the current learning rate
            lr_before_update = float(current_lr) if isinstance(current_lr, (int, float)) else current_lr.item()

            # Execute the loss function
            # execute the loss function
            # execute the loss function
            (lvalue, ntoks), grad = self.loss_function(self.model, input_tensor, target_tensor, lengths)


            # Update the optimizer
            self.optimizer.update(self.model, grad)

            # Set the batch end time
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time

            return {
                "loss": lvalue if isinstance(lvalue, (float, int)) else lvalue.item(),
                "ntoks": ntoks,
                "expected_tokens": expected_tokens,
                "batch_time": batch_time,
                "lr_before_update": lr_before_update
            }

        except Exception as e:
            logger.error(f"Error in batch {batch_index}: {e}")
            return {
                "loss": 0,
                "ntoks": 0,
                "expected_tokens": expected_tokens,
                "batch_time": time.time() - batch_start_time,
                "lr_before_update": lr_before_update
            }

    def update_progress_bar(self, batch_progress, batch_index, batch_metrics):
        # check if we need to update the progress for the batch
        if batch_index % self.progress_interval == 0:
            # calculate tokens per second
            actual_tokens_per_second = batch_metrics["ntoks"] / batch_metrics["batch_time"]
            theoretical_tokens_per_second = batch_metrics["expected_tokens"] / batch_metrics["batch_time"]

            # update the batch progress
            batch_progress.update(self.progress_interval)

            # update the stats on the progress
            batch_progress.set_postfix({
                "Batch Loss": batch_metrics["loss"],
                "Batch Tokens": batch_metrics["ntoks"],
                "Batch Time": f"{batch_metrics['batch_time']:.3f}s",
                "Tokens/s": f"{actual_tokens_per_second:.2f}",
                "LR": f"{batch_metrics['lr_before_update']:.7f}"
            })

    def log_epoch_metrics(self, epoch, num_epochs, batch_count, epoch_loss, epoch_tokens, average_batch_time, actual_tokens_per_second, theoretical_tokens_per_second):
        if batch_count > 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Loss: {epoch_loss / batch_count:.4f}")
        else:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. No batches processed.")

        logger.info(f"Epoch Tokens: {epoch_tokens}")
        logger.info(f"Average Batch Time: {average_batch_time:.3f}s")
        logger.info(f"Tokens/s: {actual_tokens_per_second:.2f} (Actual) / {theoretical_tokens_per_second:.2f} (Theoretical)")

    def checkpoint_if_necessary(self, epoch):
        if self.checkpoint_freq is not None:
            logger.info(f'Checkpointing at end of epoch {epoch+1}')
            self.save_checkpoint(f'epoch_{epoch+1}')