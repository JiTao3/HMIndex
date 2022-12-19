import socketserver
import struct
import torch
from model import LeoModel

leoModelPath = "/home/jitao/leo_index_pointnet/log/osm_en_us/osm_ne_us_1.pt"
leoModelPath = "/home/jitao/leo_index_pointnet/log/tiger/tiger_1.pt"
leoModelPath = "/home/jitao/leo_index_pointnet/log/skewed/skewed_1.pt"
leoModelPath = "/home/jitao/leo_index_pointnet/log/uniform/uniform_1.pt"
# leoModelPath = ""
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


class MetaTCPHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        leo_model = torch.load(leoModelPath, map_location=device)
        while True:
            self.data = self.request.recv(8129).strip()
            if not self.data:
                print("The client disconnects actively!")
                break
            sendData = self._leo_cal_weight_(leo_model, 128)
            self.return_client(sendData)

    def return_client(self, sendData):  # 数据处理方法
        self.request.sendall(sendData)

    def _leo_cal_weight_(self, leo_model, batchSize):
        sampledData = struct.unpack("%sd" % 1024, self.data)
        # print(sampledData)
        print("receive data!")
        input = torch.FloatTensor(sampledData).to(device)
        input = input.view(1, input.shape[-1], 1)
        weight = leo_model.encode_decode(input)
        weight = weight.view(weight.shape[-1]).tolist()
        # print(weight)
        sendData = struct.pack("%sd" % len(weight), *weight)
        return sendData


if __name__ == "__main__":
    HOST, PORT = "localhost", 12333
    # Create the server, binding to localhost on port 9999
    server = socketserver.ThreadingTCPServer((HOST, PORT), MetaTCPHandler)
    # 实例化一个多线程TCPServer
    print("Wait client . . .")
    server.serve_forever()
