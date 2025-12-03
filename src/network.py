import socket
import json

class MocapSender:
    def __init__(self, host='127.0.0.1', port=9000):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data):
        """
        Send dictionary data as JSON via UDP.
        """
        try:
            message = json.dumps(data).encode('utf-8')
            self.sock.sendto(message, (self.host, self.port))
        except Exception as e:
            print(f"Error sending data: {e}")

    def close(self):
        self.sock.close()