import mlx.core as mx
import time

class Trainer:
    def __init__(self, model, optimizer, loss_function):
        # set the model
        self.model = model

        # set the optimizer
        self.optimizer = optimizer

        # set the loss function
        self.loss_function = loss_function

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

            # load each batch file
            for input_tensor, target_tensor, lengths in batch_dataset:
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

            # end the epoch timers
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            # update the batch timers
            total_batch_time += sum(batch_times)
            total_epoch_time += epoch_time

            # show the epoch stats
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Tokens: {epoch_tokens}")

        # calculate the total and average batch times
        avg_total_batch_time = total_batch_time / (num_epochs * len(batch_dataset))
        avg_total_epoch_time = total_epoch_time / num_epochs

        # print out the total and average batch times
        print(f"Total Batch Time: {total_batch_time:.2f}s")
        print(f"Total Epoch Time: {total_epoch_time:.2f}s")
        print(f"Average Batch Time: {avg_total_batch_time:.4f}s")
        print(f"Average Epoch Time: {avg_total_epoch_time:.4f}s")