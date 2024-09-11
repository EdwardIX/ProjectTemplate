import os
import socket
import threading
import time

import pickle
import json
import subprocess as sp

SOCKET_BUFFER_SIZE = 4096

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
    def __init__(self, path):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.runners = []
        
        self.socket.bind(("0.0.0.0", 0))
        ip = get_local_ip()
        port = self.socket.getsockname()[1]

        self.socket.listen()
        print(f"[LISTENING] Server is listening on {ip}:{port}")

        with open(os.path.join(path, "connection.txt"), "w") as f:
            print(f"ip:{ip}\nport:{port}", file=f)
        
        self.listener = threading.Thread(target=self.listen_runners) # Create a thread to listen runners
        self.listener.start()

    def listen_runners(self):
        while True:
            conn, addr = self.socket.accept()
            self.runners.append({'socket': conn, 'status': RunnerStatus()})
            print(f"[NEW CONNECTION] {addr} connected.")
            threading.Thread(target=self.handle_runner, args=(len(self.runners)-1, conn)).start()

    def handle_runner(self, idx, conn):

        def handle_message(message):
            message = pickle.loads(message)
            if message['type'] == "Status":
                self.runners[idx]['status'].unpack(message['data'])
                print(self.runners[idx]['status'].hostname, self.runners[idx]['status'].gpucnt, self.runners[idx]['status'].gpumem)
            else:
                print(message)

        connected = True
        while connected:
            try:
                message = conn.recv(SOCKET_BUFFER_SIZE)
                if message:
                    handle_message(message)
                else:
                    connected = False
            except ConnectionResetError:
                connected = False

class SocketRunner:
    def __init__(self, path):
        ip, port = None, None
        with open(os.path.join(path, "connection.txt"), "r") as f:
            for l in f.readlines():
                if l.startswith("ip:"): ip = l[3:].strip()
                if l.startswith("port:"): port = int(l[5:].strip())

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((ip, port))
        print(f"[CONNECTED] ip:{ip}, port:{port}")
        self.socket_wlock = threading.Lock()

        self.threadreport = threading.Thread(target=self.send_report)
        self.threadreport.start()
    
    def send_to_server(self, data):
        with self.socket_wlock:
            self.socket.send(data)

    def send_report(self):
        status = RunnerStatus()
        while True:
            status.update()
            message = pickle.dumps({
                'type': "Status",
                'data': status.pack()
            })
            self.send_to_server(message)
            time.sleep(1)

def runserv():
    serv = SocketServer("../../../runs")
    while True:
        time.sleep(1)
        if len(serv.runners):
            print(serv.runners[0]['status'].hostname)

def runrner():
    runner = SocketRunner("../../../runs")
    runner.threadreport.join()

if __name__ == "__main__":
    runrner()