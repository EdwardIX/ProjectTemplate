import os
import socket
import threading
import time

import pickle
import json
import subprocess as sp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .server import Server
    from .runner import Runner

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('10.255.255.255', 1))
    return s.getsockname()[0]

def send_message(sock:socket.socket, message):
    message_length = len(message)
    message_length_bytes = message_length.to_bytes(4, byteorder='big')
    try:
        sock.sendall(message_length_bytes)
        sock.sendall(message)
        return True
    except Exception as e:
        print("[SEND MESSAGE ERROR]:", e)
        return False

def recv_message(sock:socket.socket):
    try:
        message_length_bytes = sock.recv(4)
    except Exception as e:
        print("[RECV MESSAGE ERROR]:", e)
        return None
    if not message_length_bytes:
        return None
    message_length = int.from_bytes(message_length_bytes, byteorder='big')
    message = sock.recv(message_length)
    return message

class SocketServer:
    def __init__(self, serv):
        self.serv:Server = serv

        # Initialize Basic Information
        self.runners = {}
        self.hostname = json.loads(sp.check_output(['gpustat', '--json']).decode())['hostname']

        # Initialize listen socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("0.0.0.0", 0))
        ip = get_local_ip()
        port = self.socket.getsockname()[1]
        self.socket.listen()
        print(f"[LISTENING] Server is listening on {ip}:{port}")
        # Save IP and Port to connection.txt
        with open(os.path.join("runs", "connection.txt"), "w") as f:
            print(f"ip:{ip}\nport:{port}", file=f)
        
        # Start Thread
        self.socket_wlock = threading.Lock() # For Simplicity, All Runner Sockets shares the same wlock.
        self.listener = threading.Thread(target=self.listen_runners) # Create a thread to listen runners
        self.listener.start()

    def listen_runners(self):
        """Thread: Handling New Incoming Runners"""
        while True:
            # Build Connection
            conn, addr = self.socket.accept()
            send_message(conn, self.hostname.encode()) # Send Hostname
            runner_name = recv_message(conn).decode() # Recv Hostname
            if runner_name in self.runners.keys():
                print(f"[REJECT CONNECTION] duplicated runners on {runner_name}")
                conn.close()
            
            # Handle New Runner
            self.runners[runner_name] = conn # Add to dict
            self.serv.on_new_runner(runner_name) # Notify Server
            print(f"[NEW CONNECTION] {runner_name} connected.")
            threading.Thread(target=self.handle_runner, args=(runner_name, conn)).start() # Start Threading

    def handle_runner(self, name, conn):
        """Thread: Handle Active Runners"""
        def handle_message(message):
            """Handle Message from Runner"""
            message = pickle.loads(message)
            if message['type'] == "RunnerStatus":
                self.serv.on_runner_status_update(name, message['data'])
            elif message['type'] == "TaskStatus":
                self.serv.on_task_status_update(message['identifier'], message['status'])
            else:
                raise RuntimeError(f"Not Recognized Message {message}")

        # Main Loop
        connected = True
        while connected:
            message = recv_message(conn)
            if message:
                handle_message(message)
            else:
                connected = False
        
        self.serv.on_del_runner(name)
        self.runners.pop(name)
        print(f"[DISCONNECT] runner {name} disconnected.")
    
    # Sending Functions ...
    def send_task(self, runner, identifier, task, gpuinfo, seed):
        message = pickle.dumps({
            'type': "RunTask",
            'identifier': identifier,
            'args': task.args, 
            'reqs': task.reqs, 
            'gpuinfo': gpuinfo,
            'seed': seed,
        })
        with self.socket_wlock:
            return send_message(self.runners[runner], message)
    
    def send_command(self, runner, command, args):
        message = pickle.dumps({
            'type': command,
            **args,
        })
        with self.socket_wlock:
            return send_message(self.runners[runner], message)

class SocketRunner:
    def __init__(self, runn):
        self.runn:Runner = runn

        # Initialize Basic Information
        self.hostname = json.loads(sp.check_output(['gpustat', '--json']).decode())['hostname']

        # Get IP and Port of Server
        ip, port = None, None
        if not os.path.exists(os.path.join("runs", "connection.txt")):
            print(f"waiting for server to start ...")
            while not os.path.exists(os.path.join("runs", "connection.txt")):
                time.sleep(1)
        with open(os.path.join("runs", "connection.txt"), "r") as f:
            for l in f.readlines():
                if l.startswith("ip:"): ip = l[3:].strip()
                if l.startswith("port:"): port = int(l[5:].strip())

        # Initialize communication socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((ip, port))
        self.connected = True
        send_message(self.socket, self.hostname.encode())
        server_name = recv_message(self.socket).decode()
        print(f"[CONNECTED] server_name:{server_name} ip:{ip}, port:{port}")

        # Start Multi Thread
        self.socket_wlock = threading.Lock()
        self.threadreport = threading.Thread(target=self.send_report)
        self.threadreport.start()
        self.threadlisten = threading.Thread(target=self.recv_server)
        self.threadlisten.start()

    def recv_server(self):
        """Thread: Listening Server Commands"""
        while True:
            message = recv_message(self.socket)
            if not message:
                self.connected = False
                break
            message = pickle.loads(message)
            
            if message['type'] == 'RunTask':
                self.runn.on_run_task(message['identifier'], message['args'], message['reqs'], message['gpuinfo'], message['seed'])
            elif message['type'] == 'StopTask':
                self.runn.on_stop_task(message['identifier'])
            else:
                raise RuntimeError(f"Not Recognized Message {message}")

    # Sending Functions ...
    def send_report(self):
        while self.connected:
            self.runn.on_update_status()
            message = pickle.dumps({
                'type': "RunnerStatus",
                'data': self.runn.status.pack()
            })
            with self.socket_wlock:
                if not send_message(self.socket, message):
                    self.connected = False
                    break
            time.sleep(1)
    
    def send_task_status(self, identifier, status):
        message = pickle.dumps({
            'type': "TaskStatus",
            'identifier': identifier, 
            'status': status,
        })
        with self.socket_wlock:
            if not send_message(self.socket, message):
                self.connected = False