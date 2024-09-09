import socket
import threading

import pickle
import json
import subprocess as sp

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('10.255.255.255', 1))
    return s.getsockname()[0]

def find_free_port():
    # 创建一个临时套接字
    temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 将套接字绑定到本地主机的一个随机端口
    temp_socket.bind(('localhost', 0))
    # 获取绑定后的端口号
    _, port = temp_socket.getsockname()
    # 关闭套接字
    temp_socket.close()
    
    return port

class RunnerStatus:
    def __init__(self):
        self.hostname = ""
        self.gpucnt = 0
        self.gpumem = []
        self.gpuusg = []

    def update(self):
        gpustat = json.loads(sp.check_output(['gpustat', '--json']).decode())
        gpus = gpustat['gpus']
        self.hostname = gpustat['hostname']
        self.gpucnt = len(gpustat)
        self.gpumem = list(map((lambda g:g['memory.total']-g['memory.used']), gpus))
        self.gpuusg = list(map((lambda g:g['utilization.gpu']), gpus))

    def pack(self):
        package = (
            self.hostname,
            self.gpucnt,
            self.gpumem,
            self.gpuusg,
        )
        return pickle.dumps(package)
    
    def unpack(self, data):
        (
            self.hostname,
            self.gpucnt,
            self.gpumem,
            self.gpuusg
        ) = pickle.loads(data)


class SocketServer:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.runners = []
        
        self.init_server_socket()

    def init_server_socket(self):
        self.socket.bind(("0.0.0.0", 0))
        ip = get_local_ip()
        port = self.socket.getsockname()[1]
        self.socket.listen()
        print(f"[LISTENING] Server is listening on {ip}:{port}")
        threading.Thread(target=self.listen_runners) # Create a thread to listen runners

    def listen_runners(self):
        while True:
            conn, addr = self.socket.accept()
            self.runners.append({'socket': conn, 'status': RunnerStatus()})
            print(f"[NEW CONNECTION] {addr} connected.")
            threading.Thread(target=self.handle_runner, args=(conn))

    def handle_runners(self, id, conn):
        connected = True
        while connected:
            # TODO

if __name__ == "__main__":
    serv = SocketServer()