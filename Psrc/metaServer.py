import socketserver
import struct


class MetaTCPHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while True:
            self.data = self.request.recv(1024).strip()
            if not self.data:
                print("The client disconnects actively!")
                break
            self.return_client()

    def return_client(self):  # 数据处理方法
        print(
            "Ip:{0} Port:{1}".format(
                self.client_address[0], self.client_address[1]
            )
        )
        print(self.data)
        # just send back the same data, but upper-cased
        sendData = struct.unpack("%sd" % 5, self.data)
        print(sendData)
        sendData = [i + 1 for i in sendData]
        sendData = struct.pack("%sd" % 5, *sendData)
        self.request.sendall(sendData)


if __name__ == "__main__":
    HOST, PORT = "localhost", 12333

    # Create the server, binding to localhost on port 9999
    server = socketserver.ThreadingTCPServer(
        (HOST, PORT), MetaTCPHandler
    )  # 实例化一个多线程TCPServer
    print("Wait client . . .")
    server.serve_forever()
