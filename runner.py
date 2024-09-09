import socket
import time

# Runner Configuration
SERVER_HOST = '192.168.245.103'  # Server IP address or hostname
SERVER_PORT = 12345
BUFFER_SIZE = 1024

def runner_client():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((SERVER_HOST, SERVER_PORT))
    print("[CONNECTED] Connected to the server.")

    try:
        while True:
            # Send status update or task completion message to the server
            message = "STATUS: Runner is running."
            client.send(message.encode('utf-8'))

            # Receive response from the server
            response = client.recv(BUFFER_SIZE).decode('utf-8')
            print(f"[SERVER RESPONSE] {response}")

            # Simulate task running time
            time.sleep(5)
    except KeyboardInterrupt:
        print("[DISCONNECT] Disconnecting from the server.")
    finally:
        client.close()

if __name__ == "__main__":
    runner_client()
