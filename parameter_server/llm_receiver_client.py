import zmq
import time
import argparse
import sys
import logging
import uuid
from datetime import datetime
from multiprocessing import Process, cpu_count, Queue
import curses

def receive_updates(client_id, server_address, server_port, queue=None, verbose=False, log=False):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    connect_address = f"tcp://{server_address}:{server_port}"
    socket.connect(connect_address)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

    start_time = datetime.now()
    updates_received = 0

    if log:
        log_filename = f"{client_id}_receiver_log.txt"
        logging.basicConfig(filename=log_filename, level=logging.INFO)
        logging.info(f"Connected to server at {connect_address}")
        logging.info(f"Client ID: {client_id}")
        logging.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"{'='*40}")

    if verbose:
        print(f"[{client_id}] Connected to server at {connect_address}")

    def listen_for_updates():
        nonlocal updates_received, start_time

        while True:
            try:
                update = socket.recv_json(flags=zmq.NOBLOCK)
                updates_received += 1
                elapsed_time = (datetime.now() - start_time).total_seconds()
                updates_per_second = updates_received / elapsed_time if elapsed_time > 0 else 0

                # Send received updates to the queue for display
                queue.put((client_id, 'progress', f"[{client_id}] Updates Received: {updates_received} | "
                                                  f"Updates Per Second: {updates_per_second:.2f}"))

                if log:
                    logging.info(f"Update Received | Updates Per Second: {updates_per_second:.2f}")

                if verbose:
                    print(f"[{client_id}] Received update: {update}")

            except zmq.Again:
                # No message received, continue to loop
                time.sleep(0.1)

    listen_for_updates()

def curses_display(stdscr, queue, client_count):
    stdscr.clear()
    client_positions = {}
    while True:
        while not queue.empty():
            client_id, message_type, message = queue.get()
            if client_id not in client_positions:
                client_positions[client_id] = len(client_positions) * 2
            pos = client_positions[client_id]
            if message_type == 'progress':
                stdscr.addstr(pos, 0, message)
            stdscr.clrtoeol()  # Clear to the end of the line to ensure a clean display
            stdscr.refresh()
        time.sleep(0.1)

def run_multiple_receivers(client_count, args):
    queue = Queue()  # Queue for inter-process communication

    processes = []
    for i in range(client_count):
        client_id = args.client_id or str(uuid.uuid4())
        p = Process(target=receive_updates, args=(
            client_id, args.server_address, args.server_port, queue, args.verbose, args.log
        ))
        p.start()
        processes.append(p)

    curses.wrapper(curses_display, queue, client_count)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a receiver client that subscribes to updates from a parameter server.")

    parser.add_argument('--client_id', type=str, help='Unique ID for the client. If not provided, a random UUID will be generated.')
    parser.add_argument('--server_address', type=str, default='localhost', help='Server address to connect to (default: localhost)')
    parser.add_argument('--server_port', type=int, default=5556, help='Server port to connect to (default: 5556)')
    parser.add_argument('--client_count', type=int, default=cpu_count(), help='Number of parallel clients to run (default: number of CPU cores)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    parser.add_argument('--log', action='store_true', help='Enable logging to a file (default: off)')

    args = parser.parse_args()

    if args.client_count > 1:
        run_multiple_receivers(args.client_count, args)
    else:
        queue = Queue()
        receive_updates(
            client_id=args.client_id or str(uuid.uuid4()),
            server_address=args.server_address,
            server_port=args.server_port,
            queue=queue,
            verbose=args.verbose,
            log=args.log
        )
