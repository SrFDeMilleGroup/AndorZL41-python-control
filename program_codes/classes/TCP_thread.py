import PyQt5.QtCore
import logging, time
import socket, selectors, struct 

# this thread handles TCP communication with another PC, it starts when this program starts
# code is from https://github.com/qw372/Digital-transfer-cavity-laser-lock/blob/8db28c2edd13c2c474d68c4b45c8f322f94f909d/main.py#L1385
class TCPThread(PyQt5.QtCore.QThread):
    update_signal = PyQt5.QtCore.pyqtSignal(dict)
    start_signal = PyQt5.QtCore.pyqtSignal()
    stop_signal = PyQt5.QtCore.pyqtSignal()

    def __init__(self, parent, host, port):
        super().__init__()
        self.parent = parent
        self.data = bytes()
        self.length_get = False
        self.host = host
        self.port = int(port)
        self.sel = selectors.DefaultSelector()

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Avoid bind() exception: OSError: [Errno 48] Address already in use
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen()
        logging.info(f"listening on: {(self.host, self.port)}")
        self.server_sock.setblocking(False)
        self.sel.register(self.server_sock, selectors.EVENT_READ, data=None)

    def run(self):
        while self.parent.tcp_active:
            events = self.sel.select(timeout=0.1)
            for key, mask in events:
                if key.data is None:
                    # this event is from self.server_sock listening
                    self.accept_wrapper(key.fileobj)
                else:
                    s = key.fileobj
                    try:
                        data = s.recv(1024) # 1024 bytes should be enough for our data
                    except Exception as err:
                        logging.error(f"TCP connection error: \n{err}")
                        data = None
                    if data:
                        self.data += data
                        while len(self.data) > 0:
                            if (not self.length_get) and len(self.data) >= 4:
                                self.length = struct.unpack(">I", self.data[:4])[0]
                                self.length_get = True
                                self.data = self.data[4:]
                            elif self.length_get and len(self.data) >= self.length:
                                message = self.data.decode('utf-8')
                                # logging.info(message)
                                if message == "Status?":
                                    # if it's just a check in message to test connection
                                    re = "Running" if self.parent.active else "Idle"
                                    try:
                                        s.sendall(re.encode('utf-8'))
                                    except Exception as err:
                                        logging.error(f"(tcp thread) Failed to reply the message. \n{err}")
                                elif message == "Stop":
                                    # if it's to stop running
                                    self.stop_signal.emit()
                                else:
                                    # if it's a message about scan sequence
                                    with open(self.parent.config["scan_file_name"], "w") as f:
                                        f.write(message)

                                    # turn on the camera here
                                    self.start_signal.emit()
                                    time.sleep(0.2)

                                    try:
                                        s.sendall("Received".encode('utf-8'))
                                    except Exception as err:
                                        logging.error(f"(tcp thread) Failed to reply the message. \n{err}")
                                t = time.time()
                                time_string = time.strftime("%Y-%m-%d  %H:%M:%S.", time.localtime(t))
                                time_string += "{:1.0f}".format((t%1)*10) # get 0.1 s time resolution
                                return_dict = {"last write": time_string}
                                self.update_signal.emit(return_dict)
                                self.data = self.data[self.length:]
                                self.length_get = False
                            else:
                                break
                    else:
                        # empty data will be interpreted as the signal of client shutting down
                        logging.info("client shutting down...")
                        self.sel.unregister(s)
                        s.close()
                        self.length_get = False
                        self.data = bytes()

        self.sel.unregister(self.server_sock)
        self.server_sock.close()
        self.sel.close()

    def accept_wrapper(self, sock):
        conn, addr = sock.accept()  # Should be ready to read
        logging.info(f"accepted connection from: {addr}")
        conn.setblocking(False)
        self.sel.register(conn, selectors.EVENT_READ, data=123) # In this application, 'data' keyword can be anything but None
        return_dict = {"client addr": addr}
        self.update_signal.emit(return_dict)
