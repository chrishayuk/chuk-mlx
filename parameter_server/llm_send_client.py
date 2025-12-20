import zmq
import time
import numpy as np
import argparse
import sys
import logging
import uuid
from datetime import datetime
from multiprocessing import Process, Queue, cpu_count
import signal
import random
from statistics import mean, stdev
import curses
import threading

# Global variables for socket and context to ensure cleanup
context = None
socket = None

def graceful_exit(signum, frame):
    global socket, context
    print("\nGracefully shutting down the client.")
    if socket:
        socket.close()
    if context:
        context.term()
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

def compute_gradients(batch_size, sequence_length):
    """Simulate the gradient computation with vectorized NumPy operations."""
    gradients = {
        "q_proj": np.random.randn(batch_size, sequence_length).mean(), 
        "k_proj": np.random.randn(batch_size, sequence_length).mean(),
        "v_proj": np.random.randn(batch_size, sequence_length).mean()
    }
    return gradients

def simulate_llm_client(client_id, batch_size, sequence_length, num_updates, update_interval, learning_rate, server_address, server_port, verbose=False, log=False, queue=None):
    global context, socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_address}:{server_port}")
    
    try:
        start_time = datetime.now()
        updates_sent = 0
        update_times = []
        tokens_per_batch = batch_size * sequence_length  # Calculate tokens per batch

        if log:
            log_filename = f"{client_id}_log.txt"
            logging.basicConfig(filename=log_filename, level=logging.INFO)
            logging.info(f"Connected to server at {server_address}:{server_port}")
            logging.info(f"Client ID: {client_id}")
            logging.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"{'='*40}")

        # Send initial connection info
        queue.put((client_id, 'connected', f"Connected to server at {server_address}:{server_port}"))
        queue.put((client_id, 'client_id', f"Client ID: {client_id}"))
        queue.put((client_id, 'start_time', f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"))

        def send_updates():
            nonlocal updates_sent, update_times, start_time

            while updates_sent < num_updates:
                time.sleep(update_interval)  # Minimize sleep to maintain throughput
                
                update_start_time = datetime.now()

                # Send and receive asynchronously to reduce latency
                socket.send_json({"type": "get"}, zmq.NOBLOCK)
                response = socket.recv_json()

                gradients = compute_gradients(batch_size, sequence_length)

                socket.send_json({
                    "type": "update",
                    "gradients": gradients,
                    "learning_rate": learning_rate,
                    "version": response["version"]
                })
                update_response = socket.recv_json()

                updates_sent += 1
                elapsed_time = (datetime.now() - start_time).total_seconds()
                updates_per_second = updates_sent / elapsed_time if elapsed_time > 0 else 0
                tokens_per_second = tokens_per_batch * updates_per_second

                # Send progress updates to the queue, including tokens processed per second
                queue.put((client_id, 'progress', f"[{client_id}] Updates Sent: {updates_sent}/{num_updates} | "
                                                  f"Updates Per Second: {updates_per_second:.2f} | "
                                                  f"Tokens Per Second: {tokens_per_second:.2f} | "
                                                  f"Current Version: {update_response['version']}"))

                if log:
                    update_time = (datetime.now() - update_start_time).total_seconds()
                    update_times.append(update_time)
                    logging.info(f"Update {updates_sent}/{num_updates} | Updates Per Second: {updates_per_second:.2f} | "
                                 f"Tokens Per Second: {tokens_per_second:.2f} | Current Version: {update_response['version']}")

                if verbose:
                    print(f"\nVerbose: Gradient Update Details: {gradients}")

        # Use threading to overlap communication and computation
        update_thread = threading.Thread(target=send_updates)
        update_thread.start()
        update_thread.join()

        if log:
            average_update_time = mean(update_times)
            update_time_stdev = stdev(update_times) if len(update_times) > 1 else 0
            logging.info(f"Average Update Time: {average_update_time:.4f} seconds")
            logging.info(f"Update Time Standard Deviation: {update_time_stdev:.4f} seconds")

    finally:
        # Ensure cleanup happens whether the loop exits normally or via an interruption
        if socket:
            socket.close()
        if context:
            context.term()

def curses_display(stdscr, queue, client_count):
    stdscr.clear()
    client_positions = {}
    while True:
        while not queue.empty():
            client_id, message_type, message = queue.get()
            if client_id not in client_positions:
                client_positions[client_id] = len(client_positions) * 4
            pos = client_positions[client_id]
            if message_type == 'connected':
                stdscr.addstr(pos, 0, message)
            elif message_type == 'client_id':
                stdscr.addstr(pos + 1, 0, message)
            elif message_type == 'start_time':
                stdscr.addstr(pos + 2, 0, message)
            elif message_type == 'progress':
                stdscr.addstr(pos + 3, 0, message)
            stdscr.clrtoeol()  # Clear to the end of the line to ensure a clean display
            stdscr.refresh()
        time.sleep(0.1)

def run_multiple_clients(client_count, args):
    queue = Queue()  # Queue for inter-process communication

    processes = []
    for i in range(client_count):
        client_id = args.client_id or str(uuid.uuid4())
        p = Process(target=simulate_llm_client, args=(
            client_id, args.batch_size, args.sequence_length, args.num_updates, args.update_interval, 
            args.learning_rate, args.server_address, args.server_port, args.verbose, args.log, queue
        ))
        p.start()
        processes.append(p)

    curses.wrapper(curses_display, queue, client_count)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate an LLM client that sends gradient updates to a parameter server.")

    parser.add_argument('--client_id', type=str, help='Unique ID for the client. If not provided, a random UUID will be generated.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of sequences processed in parallel (default: 64)')
    parser.add_argument('--sequence_length', type=int, default=128, help='Length of each sequence in tokens (default: 128)')
    parser.add_argument('--num_updates', type=int, default=100, help='Number of gradient updates to send (default: 100)')
    parser.add_argument('--update_interval', type=float, default=0.05, help='Time interval between updates in seconds (default: 0.05)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate used for gradient updates (default: 0.1)')
    parser.add_argument('--server_address', type=str, default='localhost', help='Server address to connect to (default: localhost)')
    parser.add_argument('--server_port', type=int, default=5555, help='Server port to connect to (default: 5555)')
    parser.add_argument('--client_count', type=int, default=cpu_count(), help='Number of parallel clients to run (default: number of CPU cores)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    parser.add_argument('--log', action='store_true', help='Enable logging to a file (default: off)')

    args = parser.parse_args()

    if args.client_count > 1:
        run_multiple_clients(args.client_count, args)
    else:
        queue = Queue()
        simulate_llm_client(
            client_id=args.client_id or str(uuid.uuid4()),
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_updates=args.num_updates,
            update_interval=args.update_interval,
            learning_rate=args.learning_rate,
            server_address=args.server_address,
            server_port=args.server_port,
            verbose=args.verbose,
            log=args.log,
            queue=queue
        )
