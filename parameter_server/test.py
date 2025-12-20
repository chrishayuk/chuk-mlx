import zmq
import time

def run_receiver_client(server_address="localhost", server_port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_address}:{server_port}")
    
    while True:
        # Request the latest parameters from the server
        socket.send_json({"type": "get"})
        response = socket.recv_json()
        
        # Print the received parameters and version
        parameters = response["parameters"]
        version = response["version"]
        print(f"Received parameters (version {version}): {parameters}")
        
        time.sleep(2)  # Wait for a bit before requesting again

if __name__ == "__main__":
    run_receiver_client()
