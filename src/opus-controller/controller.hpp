// controller.hpp
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>

#define PP_INTERVAL 1000000
#define PP_BACKEND_INTERVAL 10

class Controller {
public:
    Controller(std::string bindIp, int reconfigDelay, bool isEmulation, int num_rails, const std::vector<int>& topoIds, std::string outputDir);
    ~Controller();
    // Starts the server, binds, and listens for connections
    int run();

private:
    // Represents a connected client
    struct Client {
        int sock;
        std::string ip;
    };

    // Message types received from clients
    enum CtrlMsgType {
        MSG_COMM,
        MSG_CONFIG_REQ,
        MSG_UNKNOWN
    };

    // worker structure
    std::vector<std::thread> workers;
    std::queue<int> socketQueue;
    std::mutex queueMtx;
    std::condition_variable queueCv;
    bool shuttingDown = false;
    void workerLoop();


    // Parses the message header to determine type
    CtrlMsgType parseHeader(const std::string& h);

    // Handles COMM messages
    void handle_comm(const std::string& body, const std::string& peerIp, int clientSock);

    // Handles CONFIG_REQ messages
    void handle_config_req(const std::string& hexId, const std::string& peerIp);

    // Handles client connection and incoming messages
    void handleClient(int clientSock);

    void processLine(const std::string& line,
                 const std::string& peerIp,
                 int clientSock);

    // Determines topology and scale-up/scale-out based on client IPs
    int determine_topo(std::string& hexId, int localRank);
    int determine_topo_pm(std::string& hexId, int localRank);
    void determine_runtime_topo(std::string& hexId, int backendId, int collectiveCounter);

    std::string buildCommand(int topoId);

    std::string bindIp;
    bool reconfig(int topoId, int rail);
    bool reconfig_all(int topoId);

    // State tracking
    std::unordered_map<std::string, std::vector<Client>> idToClients;
    std::unordered_map<std::string, int> idToCommSize;
    std::unordered_map<std::string, std::unordered_map<int, int>> readyCount; // hexId->collectiveCounter->count

    // mapping
    std::unordered_map<std::string, std::pair<int, int>> idToTopoId; // <topoId, rail_id>

    // pre-specifiy topoIds here. 0 for big ring, 1 for east-west, 2 for north-south
    std::vector<int> topoIds; //rail_id to topoId mapping
    int num_rails;
    int cur_topoId = -1;

    // Environment setup
    int num_nodes = 1;
    int num_ranks_per_node = 1;
    std::vector<std::string> server_ips; // ipv4 addresses of all nodes

    // Synchronization
    std::mutex mapMtx;
    // std::mutex clientsMtx;

    // Listening port
    static constexpr int PORT = 1234;

    const std::string POLATIS_HOST = "10.253.51.12";
    const std::string POLATIS_PORT = "830";
    const std::string POLATIS_USER = "admin";
    const std::string POLATIS_PASS = "root";
    const std::string POLATIS_CONFIG_DIR = "config";

    int reconfigDelay = 0;

    // mode
    bool isEmulation = false;

    // output directory for logs
    std::string outputDir = "./logs";
};
