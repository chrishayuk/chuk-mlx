import mlx.core as mx
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, loss_function, progress_interval=1):
        # set the model
        self.model = model

        # set the optimizer
        self.optimizer = optimizer

        # set the loss function
        self.loss_function = loss_function
        
        # set the progress update interval
        self.progress_interval = progress_interval

    def train(self, num_epochs, batch_dataset):
        # set the timers
        total_batch_time = 0
        total_epoch_time = 0

        # loop through the epochs
        for epoch in range(num_epochs):
            # kick off the timer for epoch
            epoch_start_time = time.time()

            # reset the batch settings
            batch_times = []
            epoch_loss = 0
            epoch_tokens = 0

            # create a progress bar for the batches
            batch_progress = tqdm(total=len(batch_dataset), desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

            # load each batch file
            for batch_index, (input_tensor, target_tensor, lengths) in enumerate(batch_dataset, start=1):
                # kick off the batch timer
                batch_start_time = time.time()

                # execute the loss function
                (lvalue, ntoks), grad = self.loss_function(self.model, input_tensor, target_tensor, lengths)

                # call the optimizer
                self.optimizer.update(self.model, grad)

                # eval
                mx.eval(self.model.parameters(), self.optimizer.state, lvalue)

                # update the loss value for the epoch
                epoch_loss += lvalue.item()
                epoch_tokens += ntoks.item()

                # end the batch timers
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)

                # update the progress bar at the specified interval
                if batch_index % self.progress_interval == 0:
                    batch_progress.update(self.progress_interval)
                    batch_progress.set_postfix({
                        "Batch Loss": lvalue.item(),
                        "Batch Tokens": ntoks.item(),
                        "Batch Time": f"{batch_time:.3f}s"
                    })

            # update the remaining progress
            remaining_batches = len(batch_dataset) % self.progress_interval
            if remaining_batches > 0:
                batch_progress.update(remaining_batches)
                batch_progress.set_postfix({
                    "Epoch Loss": epoch_loss / len(batch_dataset),
                    "Epoch Tokens": epoch_tokens,
                    "Avg Batch Time": f"{sum(batch_times) / len(batch_times):.3f}s"
                })

            # end the epoch timers
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            # update the batch timers
            total_batch_time += sum(batch_times)
            total_epoch_time += epoch_time