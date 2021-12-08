#include <iostream>
#include <cstring>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;

int connectMetaServer(vector<float> &modelParam, vector<double> &sampleData, int port, int modelParamSize);
