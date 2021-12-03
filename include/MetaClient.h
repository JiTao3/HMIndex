#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;

int connectMetaServer(int port)
{
    int ret, conn_fd;
    int client = socket(AF_INET, SOCK_STREAM, 0);
    if (client == -1)
    {
        cout << "Error socket" << endl;
        return 0;
    }
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (connect(client, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0)
    {
        cout << "Error connect" << endl;
        return 0;
    }
    cout << "...connect" << endl;
    double data[5] = {1, 2, 3, 4, 5};
    double recvData[5];
    send(client, data, sizeof(data), 0);

    memset(recvData, 0, sizeof(recvData));
    int len = recv(client, recvData, sizeof(recvData), 0);
    cout << recvData[0] << endl;
    cout << recvData[1] << endl;
    cout << recvData[2] << endl;
    cout << recvData[3] << endl;
    cout << recvData[4] << endl;
    close(client);
}