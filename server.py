import socket
import threading

# Server Configuration
SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 12345
BUFFER_SIZE = 1024

# Dictionary to keep track of connected runners
runners = {}

def handle_runner(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    while connected:
        try:
            message = conn.recv(BUFFER_SIZE).decode('utf-8')
            if message:
                print(f"[{addr}] {message}")
                # Parse the message and decide the action here (e.g., task assignment)
                # For simplicity, echoing the message back to the runner
                conn.send("ACK".encode('utf-8'))
            else:
                # Client closed the connection
                connected = False
        except ConnectionResetError:
            connected = False
    print(f"[DISCONNECT] {addr} disconnected.")
    conn.close()

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((SERVER_HOST, SERVER_PORT))
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER_HOST}:{SERVER_PORT}")
    
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_runner, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")

if __name__ == "__main__":
    start_server()
