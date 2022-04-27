#include "MetaClient.h"

int connectMetaServer(vector<float> &modelParam, vector<double> &sampleData, int port, int modelParamSize)
{
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

    double data[sampleData.size()];
    double recvData[modelParamSize];
    for (int i = 0; i < sampleData.size(); i++)
    {
        data[i] = sampleData[i];
        // cout << data[i] << " ";
    }
    // cout << endl;
    send(client, data, sizeof(data), 0);
    memset(recvData, 0, sizeof(recvData));
    int len = recv(client, recvData, sizeof(recvData), 0);
    for (int i = 0; i < modelParamSize; i++)
        modelParam.push_back(recvData[i]);
    close(client);
    return port;
}