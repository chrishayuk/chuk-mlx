import zmq

def run_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # REQ socket for synchronous communication
    socket.connect("tcp://localhost:5555")  # Connect to the server

    # Request current parameters
    socket.send_json({"type": "get"})
    response = socket.recv_json()
    print(f"Received Parameters: {response['parameters']}")
    print(f"Parameter Version: {response['version']}")

    # Send gradient update
    gradients = {"q_proj": 0.01, "k_proj": 0.02}
    socket.send_json({
        "type": "update",
        "gradients": gradients,
        "learning_rate": 0.1,
        "version": response["version"]
    })
    update_response = socket.recv_json()

    if update_response["success"]:
        print(f"Update successful. New Version: {update_response['version']}")
    else:
        print("Update failed due to version mismatch.")

if __name__ == "__main__":
    run_client()
