import zmq
import numpy as np

class ParameterServer:
    def __init__(self):
        self.parameters = {"q_proj": 0.1, "k_proj": 0.2, "v_proj": 0.3}  # Example parameters
        self.version = 0

    def get_parameters(self):
        return self.parameters, self.version

    def update_parameters(self, gradients, learning_rate, version):
        if version == self.version:
            for name, grad in gradients.items():
                if name in self.parameters:
                    self.parameters[name] -= learning_rate * grad
            self.version += 1
            return True, self.version
        else:
            return False, self.version

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP socket for synchronous communication
    socket.bind("tcp://*:5555")  # Bind the server to port 5555

    ps = ParameterServer()

    while True:
        # Wait for the next request from a client
        message = socket.recv_json()

        if message["type"] == "get":
            # Handle parameter retrieval
            parameters, version = ps.get_parameters()
            socket.send_json({"parameters": parameters, "version": version})
        
        elif message["type"] == "update":
            # Handle parameter update
            success, new_version = ps.update_parameters(message["gradients"], message["learning_rate"], message["version"])
            socket.send_json({"success": success, "version": new_version})

if __name__ == "__main__":
    run_server()
