// conntroller.cpp
#include "controller.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <sys/un.h>
#include <netdb.h>

#include <iostream>
#include <cstring>
#include <thread>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <netinet/tcp.h>
#include <filesystem>
#include <cstdlib>

namespace {
std::string command_quote(const std::string& value) {
    return "\"" + value + "\"";
}

int env_port(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == 0) return fallback;
    try {
        int port = std::stoi(value);
        if (port < 1 || port > 65535) throw std::out_of_range("port");
        return port;
    } catch (const std::exception&) {
        std::cerr << "[STATUS] Invalid " << name << "; using " << fallback << std::endl;
        return fallback;
    }
}
}

bool send_all(int sock, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(sock, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}


Controller::Controller(std::string bindIp, int reconfigDelay, bool isEmulation, int num_rails, const std::vector<int>& topoIds, std::string outputDir, std::string controllerDir) : bindIp(bindIp), topoIds(topoIds), num_rails(num_rails), reconfigDelay(reconfigDelay), isEmulation(isEmulation), outputDir(outputDir), controllerDir(controllerDir) {
    // Read environment variables
    const char* numNodesEnv = std::getenv("NUM_NODES");
    const char* numRanksPerNodeEnv = std::getenv("NUM_RANKS_PER_NODE");
    const char* serverIpsEnv = std::getenv("SERVER_IPS");

    num_nodes = numNodesEnv ? std::stoi(numNodesEnv) : 1;
    num_ranks_per_node = numRanksPerNodeEnv ? std::stoi(numRanksPerNodeEnv) : 1;
    controllerPort = env_port("OPUS_CONTROLLER_PORT", controllerPort);
    if (const char* value = std::getenv("OPUS_IPC_DIR")) {
        if (*value != 0) ipcDir = value;
    }
    if (const char* value = std::getenv("OPUS_IPC_PREFIX")) {
        if (*value != 0) ipcPrefix = value;
    }
    if (ipcPrefix.empty()) {
        ipcPrefix = "opus_controller_ipc_" + std::to_string(static_cast<long long>(getpid()));
    }

    std::error_code fsError;
    std::filesystem::create_directories(outputDir, fsError);
    if (fsError) {
        std::cerr << "[STATUS] ERROR: Cannot create output directory " << outputDir
                  << ": " << fsError.message() << std::endl;
        return;
    }

    std::vector<std::string> server_domains;

    if (serverIpsEnv) {
        std::istringstream ss(serverIpsEnv);
        std::string domain;
        while (std::getline(ss, domain, ',')) {
            server_domains.push_back(domain);
        }
    }

    // Translate domain names to IP addresses using DNS lookup
    for (const auto& domain : server_domains) {
        struct addrinfo hints{}, *res = NULL;
        hints.ai_family = AF_INET; // IPv4
        hints.ai_socktype = SOCK_STREAM;

        if (getaddrinfo(domain.c_str(), nullptr, &hints, &res) == 0) {
            sockaddr_in* addr = reinterpret_cast<sockaddr_in*>(res->ai_addr);
            char ipStr[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &addr->sin_addr, ipStr, sizeof(ipStr));
            server_ips.push_back(ipStr);
            freeaddrinfo(res);
        } else {
            std::cerr << "[STATUS] ERROR: Failed to resolve domain: " << domain << std::endl;
        }
    }
    // Print the resolved server IPs
    std::cout << "[STATUS] Resolved server IPs:" << std::endl;
    for (const auto& ip : server_ips) {
        std::cout << " - " << ip << std::endl;
    }

    // Start one Python worker per rail using paths relative to the executable.
    std::string configScript = (std::filesystem::path(controllerDir) / "config.py").string();
    std::string pythonCommand = "OPUS_IPC_DIR=" + command_quote(ipcDir) +
                                " OPUS_IPC_PREFIX=" + command_quote(ipcPrefix) +
                                " python3 -u " + command_quote(configScript) +
                                (isEmulation ? " -e " : " ") +
                                std::to_string(reconfigDelay) +
                                " " + std::to_string(num_rails);
    std::cout << "[STATUS] Starting Python process: " << pythonCommand << std::endl;

    for (int i = 0; i < num_rails; i++) {
        std::string logPath = (std::filesystem::path(outputDir) /
                               ("python_output_rail_" + std::to_string(i) + ".log")).string();
        std::string command = pythonCommand + " " + std::to_string(i) + " > " +
                              command_quote(logPath) + " 2>&1 &";
        int ret = std::system(command.c_str());

        if (ret != 0) {
            std::cerr << "[STATUS] ERROR: Failed to execute Python script" << std::endl;
            return;
        }
    }

    // The IPC connection below retries while the Python rail worker starts.
    // TODO
    bool success = reconfig_all(0);
    if (!success) {
        std::cerr << "[STATUS] Initial network configuration failed\n";
    } else {
        // cur_topoId = topoIds[0];
        std::cout << "[STATUS] Initial network configuration applied\n";
    }


    // start worker pool
    int numThreads = std::max(2u, std::thread::hardware_concurrency());

    std::cout << "[STATUS] Starting thread pool with "
            << numThreads << " workers\n";

    for (int i = 0; i < numThreads; ++i) {
        workers.emplace_back(&Controller::workerLoop, this);
    }
}

Controller::~Controller()
{
    {
        std::lock_guard<std::mutex> lk(queueMtx);
        shuttingDown = true;
    }
    queueCv.notify_all();

    for (auto& t : workers)
        t.join();
}


Controller::CtrlMsgType Controller::parseHeader(const std::string& h) {
    if      (h == "COMM")       return MSG_COMM;
    else if (h == "CONFIG_REQ") return MSG_CONFIG_REQ;
    return MSG_UNKNOWN;
}

int Controller::determine_topo(std::string& hexId, int localRank) {
    // return domain, determine topoId for the current group

    std::hash<std::string> hasher;
    size_t hashValue = hasher(hexId);
    int colorCode = static_cast<int>(hashValue % 7) + 31; // ANSI color codes (31-37 for foreground colors)

    int domain = 0; // 0 is scale-out, 1 is scale-up
    const auto& clients = idToClients[hexId];

    std::vector<int> addrLastByte;
    for (const auto& client : clients) {
        auto lastDot = client.ip.find_last_of('.');
        if (lastDot == std::string::npos) {
            std::cerr << "[STATUS] Malformed IP address: " << client.ip << std::endl;
            return -1;
        }
        int lastByte = std::stoi(client.ip.substr(lastDot + 1));
        addrLastByte.push_back(lastByte);
    }

    std::sort(addrLastByte.begin(), addrLastByte.end());

    bool sequential = true;
    int prevLastByte = -1;
    int size = 0; // num nodes
    
    bool sameDomain = true;

    for (const auto& lastByte : addrLastByte) {
        if (lastByte != prevLastByte) {
            size++;
        }
        prevLastByte = lastByte;
    }

    for (const auto& lastByte : addrLastByte) {
        if (prevLastByte != -1 && lastByte != prevLastByte) {
            sameDomain = false;
            break;
        }
        prevLastByte = lastByte;
    }

    prevLastByte = -1;
    for (const auto& lastByte : addrLastByte) {
        if (prevLastByte != -1 && lastByte > prevLastByte + 1) {
            sequential = false;
            break;
        }
        prevLastByte = lastByte;
    }

    // determine domain
    if (sameDomain) {
        std::cout << "\033[" << colorCode << "m[STATUS] IPs are in the SAME DOMAIN for commId=" << hexId << "\033[0m" << std::endl;
        domain = 1; // scale-up
    } else {
        std::cout << "\033[" << colorCode << "m[STATUS] IPs are in DIFFERENT DOMAINS for commId=" << hexId << "\033[0m" << std::endl;
        domain = 0; // scale-out
    }

    std::cout << "\033[" << colorCode << "m[STATUS] number of host: " << size << " for commId=" << hexId << "\033[0m" << std::endl;
    // if (size > 2) {
    //     // TODO
    //     idToTopoId[hexId] = 0;
    // } else {
    if (localRank >= num_rails) {
        throw std::runtime_error("localRank exceeds number of rails");
    }
    if (sequential) {
        std::cout << "\033[" << colorCode << "m[STATUS] IPs are sequential for commId=" << hexId << "\033[0m" << std::endl;
        idToTopoId[hexId] = {1, localRank};
    } else {
        std::cout << "\033[" << colorCode << "m[STATUS] IPs are NOT sequential for commId=" << hexId << "\033[0m" << std::endl;
        idToTopoId[hexId] = {2, localRank};
    }
    // }
    
    return domain;
}

int Controller::determine_topo_pm(std::string& hexId, int localRank) {
    // return domain, determine topoId for the current group
    // TODO get current list of IP address. And decide client IP locations

    std::hash<std::string> hasher;
    size_t hashValue = hasher(hexId);
    int colorCode = static_cast<int>(hashValue % 7) + 31; // ANSI color codes (31-37 for foreground colors)

    int domain = 0; // 0 is scale-out, 1 is scale-up
    const auto& clients = idToClients[hexId];
    int numClients = clients.size();

    std::vector<int> clientIpIdxs;

    // Locate the index of client.ip in server_ips
    for (const auto& client : clients) {
        auto it = std::find(server_ips.begin(), server_ips.end(), client.ip);
        if (it != server_ips.end()) {
            clientIpIdxs.push_back(std::distance(server_ips.begin(), it));
        } else {
            std::cerr << "[STATUS] ERROR: Client IP " << client.ip << " not found in server_ips" << std::endl;
            return -1;
        }
    }

    std::sort(clientIpIdxs.begin(), clientIpIdxs.end());

    bool sequential = true;
    int prevIdx = -1;
    int numNode = 1; // num nodes
    bool sameDomain = true;

    for (const auto& idx : clientIpIdxs) {
        if (idx != prevIdx && prevIdx != -1) {
            numNode++;
            sameDomain = false;

            if (idx != prevIdx + 1) {
                sequential = false;
            }
        }
        prevIdx = idx;
    }

    // determine domain
    if (sameDomain) {
        std::cout << "\033[" << colorCode << "m[STATUS] IPs are in the SAME DOMAIN for commId=" << hexId << "\033[0m" << std::endl;
        domain = 1; // scale-up
    } else {
        std::cout << "\033[" << colorCode << "m[STATUS] IPs are in DIFFERENT DOMAINS for commId=" << hexId << "\033[0m" << std::endl;
        domain = 0; // scale-out
    }

    std::cout << "\033[" << colorCode << "m[STATUS] number of host: " << numNode << " for commId=" << hexId << "\033[0m" << std::endl;


    if (localRank >= num_rails) {
        throw std::runtime_error("localRank exceeds number of rails");
    }
    if (sequential) {
        std::cout << "\033[" << colorCode << "m[STATUS] IPs are sequential for commId=" << hexId << "\033[0m" << std::endl;
        idToTopoId[hexId] = {1, localRank};
    } else {
        std::cout << "\033[" << colorCode << "m[STATUS] IPs are NOT sequential for commId=" << hexId << "\033[0m" << std::endl;
        idToTopoId[hexId] = {2, localRank};
    }
    // }
    
    return domain;
}


void Controller::handle_comm(const std::string& body, const std::string& peerIp, int clientSock) {
    // Body is "<hexId>,<nranks>,<localrank>"
    auto comma = body.find(',');
    if (comma == std::string::npos) {
        std::cerr << "\033[" << 31 << "m[RECV] Malformed COMM body: " << body << "\033[0m\n";
        return;
    }
    std::string hexId = body.substr(0, comma);

    auto secondComma = body.find(',', comma + 1);
    if (secondComma == std::string::npos) {
        std::cerr << "\033[" << 31 << "m[RECV] Malformed COMM body: " << body << "\033[0m\n";
        return;
    }
    int nr = std::stoi(body.substr(comma + 1, secondComma - comma - 1));
    int localRank = std::stoi(body.substr(secondComma + 1));

    std::hash<std::string> hasher;
    size_t hashValue = hasher(hexId);
    int colorCode = static_cast<int>(hashValue % 7) + 31; // ANSI color codes (31-37 for foreground colors)

    {
        std::lock_guard<std::mutex> lk(mapMtx);
        // On first time, record expected group size
        auto itSize = idToCommSize.find(hexId);
        if (itSize == idToCommSize.end()) {
            idToCommSize[hexId] = nr;
        } else if (itSize->second != nr) {
            std::cerr << "\033[" << colorCode << "m[RECV] WARNING: COMM for " << hexId
                    << " repeated with different nranks: "
                    << itSize->second << " vs " << nr << "\033[0m\n";
        }

        // Track this client’s IP
        if (idToClients.find(hexId) == idToClients.end()) {
            idToClients[hexId] = std::vector<Client>();
        }
        idToClients[hexId].push_back({clientSock, peerIp});

        std::cout << "\033[" << colorCode << "m[RECV] COMM" 
                << "\n ip=" << peerIp
                << "\n commId=" << hexId
                << "\n registered sz=" << idToClients[hexId].size()
                << "\n total sz=" << idToCommSize[hexId] << "\033[0m" << std::endl;

        if (idToClients[hexId].size() == idToCommSize[hexId]) {
            std::cout << "\033[" << colorCode << "m\n [STATUS] !!!!!!ALL REGISTERED for commId=" << hexId << "\033[0m" << std::endl;
            
            // TODO: select multipe railId based on the localRank count
            // TODO: select correct topoID even when natural IP assignment is not sequential
            // int domain = determine_topo(hexId, localRank);
            int domain = determine_topo_pm(hexId, localRank);

            // TODO: think about topology for comm across multiple rails
            // auto [topoId, railId] = idToTopoId[hexId];

            // Apply network reconfiguration
            // bool rc = reconfig(topoId);
            // if (!rc) {
            //     std::cerr << "\033[" << colorCode << "m[STATUS] ERROR applying network reconfiguration for commId=" 
            //             << hexId <<  "\033[0m" << std::endl;
            //     return;
            // }
                // cur_topoId = topoId;
        
            // broadcast domain to all clients
            auto itClients = idToClients.find(hexId);
            for (auto &c : itClients->second) {
                std::string ack = "CHECKIN_ACK;" + hexId + "," + std::to_string(domain) + "\n";
                send_all(c.sock, ack.data(), ack.size());
                std::cout << "\033[" << colorCode << "m[SENT] CHECKIN_ACK" 
                        << "\nip=" << c.ip
                        << "\ncommId=" << hexId << "\033[0m" << std::endl;
            }
        }
    }

}

// std::string Controller::buildCommand(int topoId) {
//     std::ostringstream oss;
//     oss << "netconf-console --host=" << POLATIS_HOST 
//         << " --port=" << POLATIS_PORT 
//         << " -u " << POLATIS_USER 
//         << " -p " << POLATIS_PASS 
//         << " --edit-config=" << "config/oxc_" << topoId << ".xml";
//     return oss.str();
// }

bool Controller::reconfig(int topoId, int rail) {
    // std::string command = buildCommand(topoId);

    const std::string IPC_SOCKET_PATH =
        (std::filesystem::path(ipcDir) / (ipcPrefix + "_" + std::to_string(rail))).string();
    sockaddr_un ipcAddr{};
    ipcAddr.sun_family = AF_UNIX;
    if (IPC_SOCKET_PATH.size() >= sizeof(ipcAddr.sun_path)) {
        std::cerr << "[STATUS] ERROR: IPC socket path is too long: " << IPC_SOCKET_PATH << std::endl;
        return false;
    }
    strncpy(ipcAddr.sun_path, IPC_SOCKET_PATH.c_str(), sizeof(ipcAddr.sun_path) - 1);

    int ipcSocket = -1;
    for (int attempt = 0; attempt < 500; ++attempt) {
        ipcSocket = socket(AF_UNIX, SOCK_STREAM, 0);
        if (ipcSocket >= 0 && connect(ipcSocket, (sockaddr*)&ipcAddr, sizeof(ipcAddr)) == 0) {
            break;
        }
        if (ipcSocket >= 0) close(ipcSocket);
        ipcSocket = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (ipcSocket < 0) {
        std::cerr << "[STATUS] ERROR: Failed to connect to IPC server at " << IPC_SOCKET_PATH << std::endl;
        return false;
    }

    // Send the topology ID to the IPC server
    std::string topoIdStr = std::to_string(topoId) + "\n";

    std::cout << "[STATUS] Sending topology ID " << topoIdStr << " to rail ctl " << std::to_string(rail) << std::endl;
    if (send(ipcSocket, topoIdStr.c_str(), topoIdStr.size(), 0) < 0) {
        std::cerr << "[STATUS] ERROR: Failed to send topology ID to IPC server" << std::endl;
        close(ipcSocket);
        return false;
    }

    // Receive acknowledgment from the IPC server
    char buffer[256];
    ssize_t bytesRead = recv(ipcSocket, buffer, sizeof(buffer) - 1, 0);
    if (bytesRead < 0) {
        std::cerr << "[STATUS] ERROR: Failed to receive acknowledgment from IPC server" << std::endl;
        close(ipcSocket);
        return false;
    }

    buffer[bytesRead] = '\0';
    std::string response(buffer);

    // Close the IPC socket
    close(ipcSocket);

    // Check the response from the IPC server
    if (response.find("SUCCESS") != std::string::npos) {
        // std::cout << "[STATUS] IPC server acknowledged configuration application" << std::endl;
        // Get the current time
        bool configApplied = response.find("CONFIG-ACK") != std::string::npos;
        auto now = std::chrono::system_clock::now();
        auto currentTime = std::chrono::system_clock::to_time_t(now);
        auto currentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        char timeBuffer[20];
        std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", std::localtime(&currentTime));
        std::cout << "[STATUS] " << timeBuffer
              << "." << std::setfill('0') << std::setw(3) << currentTimeMs.count()
              << ", new topo: " << (configApplied ? "yes" : "no")
              << ", topoId: " << topoId << std::endl;

        return true;
    } else {
        std::cerr << "[STATUS] ERROR: IPC server failed to apply configuration" << std::endl;
        return false;
    }
}

bool Controller::reconfig_all(int topoId) {
    bool all_success = true;
    for (int rail = 0; rail < num_rails; rail++) {
        bool rc = reconfig(topoId, rail);
        if (!rc) {
            all_success = false;
            std::cerr << "[STATUS] ERROR applying network reconfiguration for rail " << rail << std::endl;
        }
    }
    return all_success;
}

int set_PP_topo(int topoId, int pp_group) {
    if (pp_group <= 0) {
        throw std::invalid_argument("pp_group must be positive");
    }

    // Flip the specified bits
    topoId |= (1 << (pp_group - 1));
    topoId |= (1 << pp_group);
    return topoId;
}

int set_non_PP_topo(int topoId, int pp_stage) {
    if (pp_stage < 0) {
        throw std::invalid_argument("pp_stage must be non-negative");
    }

    topoId &= ~(1 << pp_stage); // links for non consecutive 1 will be dangling
    return topoId;
}

void Controller::determine_runtime_topo(std::string& hexId, int backendId, int collectiveCounter) {
    int rail = idToTopoId[hexId].second;
    if (backendId > 10) {
        // PP
        int pp_size = idToCommSize[hexId];
        int pp_group = backendId / PP_BACKEND_INTERVAL;
        if (collectiveCounter / PP_INTERVAL != pp_group) {
            throw std::runtime_error("PP backendId and collectiveCounter mismatch");
        }
        topoIds[rail] = set_PP_topo(topoIds[rail], pp_group);

    } else {
        int pp_stage = collectiveCounter / PP_INTERVAL;
        topoIds[rail] = set_non_PP_topo(topoIds[rail], pp_stage);
    }
}

bool all_arrive(int backendId, int cnt, int size) {
    // for PP, we wait for all ranks to arrive
    if (backendId > PP_BACKEND_INTERVAL) {
        return cnt >= 2;
    } else {
        // for non-PP, we wait for half of the ranks to arrive
        return cnt >= size;
    }
}


void Controller::handle_config_req(const std::string& body, const std::string& peerIp) {
    int cnt = 0;
    auto firstComma = body.find(',');
    if (firstComma == std::string::npos) {
        std::cerr << "[RECV] Malformed CONFIG_REQ body: " << body << "\n";
        return;
    }
    std::string hexId = body.substr(0, firstComma);

    auto secondComma = body.find(',', firstComma + 1);
    if (secondComma == std::string::npos) {
        std::cerr << "[RECV] Malformed CONFIG_REQ body: " << body << "\n";
        return;
    }
    int backendId = std::stoi(body.substr(firstComma + 1, secondComma - firstComma - 1));
    int collectiveCounter = std::stoi(body.substr(secondComma + 1));

    std::hash<std::string> hasher;
    size_t hashValue = hasher(hexId);
    int colorCode = static_cast<int>(hashValue % 7) + 31; // ANSI color codes (31-37 for foreground colors)


    auto itClients = idToClients.find(hexId);
    auto itSize    = idToCommSize.find(hexId);
    if (itClients == idToClients.end() || itSize == idToCommSize.end()) {
        std::cout << "[RECV] Unexpected CONFIG_REQ from " << peerIp
                << " for commId=" << hexId << "\n";
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mapMtx);
        readyCount[hexId][collectiveCounter]++;
        cnt = readyCount[hexId][collectiveCounter];
    }

    auto now = std::chrono::system_clock::now();
    auto currentTime = std::chrono::system_clock::to_time_t(now);
    auto currentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    char timeBuffer[20];
    std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", std::localtime(&currentTime));

    std::cout << "\033[" << colorCode << "m[RECV] CONFIG_REQ, " 
              << timeBuffer
              << "." << std::setfill('0') << std::setw(3) << currentTimeMs.count()
              << "\nip=" << peerIp
              << "\ncommId=" << hexId
              << ", backendId=" << backendId
              << ", collectiveCounter=" << collectiveCounter
              << ", cnt/size: " << cnt << "/" << itSize->second << "\033[0m" << std::endl;

    if (all_arrive(backendId, cnt, itSize->second)) {
        // std::cout << "\n ------------------ " << std::endl;
        int inflight_collective = 0;
        {
            std::lock_guard<std::mutex> lk(mapMtx);
            readyCount[hexId].erase(collectiveCounter);
            inflight_collective = readyCount[hexId].size();
        }

        determine_runtime_topo(hexId, backendId, collectiveCounter);
        int railId = idToTopoId[hexId].second;
        int topoId = topoIds[railId];
        
        std::cout << "\033[" << colorCode << "m\n[STATUS] !!!!!!ALL READY for commId=" << hexId 
            << ", backendId=" << backendId
            << ", collectiveCounter=" << collectiveCounter 
            << ", TopoId=" << topoId
            << ", railId=" << railId
            << ", inflight_collective=" << inflight_collective
            << "\033[0m" << std::endl;

        // Apply network reconfiguration
        bool rc = reconfig(topoId, railId);
        if (!rc) {
            std::cerr << "\033[" << colorCode << "m[STATUS] ERROR applying network reconfiguration for commId=" 
                    << hexId <<  "\033[0m" << std::endl;
            return;
        }

        for (auto &c : itClients->second) {
            std::string ack = "CONFIG_ACK;" + hexId + "\n";
            send_all(c.sock, ack.data(), ack.size());
            std::cout << "\033[" << colorCode << "m[SENT] CONFIG_ACK"
                    << "\nip=" << c.ip
                    << "\ncommId=" << hexId
                    << ", backendId=" << backendId
                    << ", collectiveCounter=" << collectiveCounter
                    << ", TopoId=" << topoId
                    << "\033[0m" << std::endl;
        }
        // Broadcast CONFIG_ACK three times
        // for (int repeat = 0; repeat < 3; ++repeat) {
        //     for (auto &c : itClients->second) {
        //         std::string ack = "CONFIG_ACK;" + hexId + "\n";
        //         send_all(c.sock, ack.data(), ack.size());
        //         std::cout << "\033[" << colorCode << "m[SENT] CONFIG_ACK ("
        //                 << (repeat + 1) << "/3)"
        //                 << "\nip=" << c.ip
        //                 << "\ncommId=" << hexId
        //                 << ", backendId=" << backendId
        //                 << ", collectiveCounter=" << collectiveCounter
        //                 << ", TopoId=" << topoId
        //                 << "\033[0m" << std::endl;
        //     }
        //     std::this_thread::sleep_for(std::chrono::milliseconds(200));
        // }
    }
}

void Controller::processLine(const std::string& line,
                             const std::string& peerIp,
                             int clientSock) {
    if (line.empty()) return;

    auto sep = line.find(';');
    std::string hdr  = (sep == std::string::npos) ? line : line.substr(0, sep);
    std::string body = (sep == std::string::npos) ? ""   : line.substr(sep + 1);

    switch (parseHeader(hdr)) {
        case MSG_COMM:
            handle_comm(body, peerIp, clientSock);
            break;
        case MSG_CONFIG_REQ:
            handle_config_req(body, peerIp);
            break;
        default:
            std::cerr << "[RECV] Unknown header '" << hdr
                      << "' from " << peerIp << "\n";
    }
}

void Controller::handleClient(int clientSock){
    sockaddr_in peerAddr{};
    socklen_t plen = sizeof(peerAddr);
    getpeername(clientSock, (sockaddr*)&peerAddr, &plen);

    char ipbuf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &peerAddr.sin_addr, ipbuf, sizeof(ipbuf));
    std::string peerIp(ipbuf);

    // Disable Nagle on client socket
    int flag = 1;
    setsockopt(clientSock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    char buf[4096];
    std::string pending;
    pending.reserve(4096);

    while (true) {
        ssize_t n = recv(clientSock, buf, sizeof(buf), 0);

        if (n == 0) {
            // client closed
            break;
        }
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("recv");
            break;
        }

        pending.append(buf, n);

        size_t pos;
        while ((pos = pending.find('\n')) != std::string::npos) {
            std::string line = pending.substr(0, pos);
            pending.erase(0, pos + 1);

            if (!line.empty() && line.back() == '\r')
                line.pop_back();

            processLine(line, peerIp, clientSock);
        }
    }

    close(clientSock);
}

void Controller::workerLoop()
{
    while (true) {
        int sock;

        {
            std::unique_lock<std::mutex> lk(queueMtx);
            queueCv.wait(lk, [&] {
                return shuttingDown || !socketQueue.empty();
            });

            if (shuttingDown && socketQueue.empty())
                return;

            sock = socketQueue.front();
            socketQueue.pop();
        }

        handleClient(sock);
    }
}

int Controller::run() {
    int ls = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(controllerPort);
    
    if (inet_pton(AF_INET, bindIp.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "Bad bind IP\n"; return 1;
    }
    if (bind(ls, (sockaddr*)&addr, sizeof(addr)) < 0 || listen(ls, SOMAXCONN) < 0) {
        perror("bind/listen"); return 1;
    }

    while (true) {
        int cs = accept(ls, nullptr, nullptr);

        if (cs < 0) { perror("accept"); continue; }
        // std::cout << "Received connection from " << inet_ntoa(addr.sin_addr)
        //           << ":" << ntohs(addr.sin_port) << "\n";
        {
            std::lock_guard<std::mutex> lk(queueMtx);
            socketQueue.push(cs);
        }
        queueCv.notify_one();
    }
    return 0;
}

int main(int argc, char* argv[]) {
    std::string bindIp = "0.0.0.0";
    int reconfigDelay = 0;
    bool isEmulation = false;
    int num_rails = 1;
    std::string outputDir = "./logs";
    std::vector<int> topoIds;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-I" && i + 1 < argc) {
            bindIp = argv[++i];
        } else if (arg == "-d" && i + 1 < argc) {
            try {
                reconfigDelay = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid reconfig_delay: " << argv[i] << "\n"; return 1;
            }
            if (reconfigDelay < 0) {
                std::cerr << "reconfig_delay must be non-negative\n"; return 1;
            }
            std::cout << "[STATUS] Reconfig delay set to " << reconfigDelay << " ms\n";
        } else if (arg == "-e") {
            isEmulation = true;
        } else if (arg == "-r" && i + 1 < argc) {
            try {
                num_rails = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid num_rails: " << argv[i] << "\n"; return 1;
            }
            if (num_rails <= 0) {
                std::cerr << "num_rails must be positive\n"; return 1;
            }
            std::cout << "[STATUS] Number of rails set to " << num_rails << "\n";
            topoIds.resize(num_rails, 0);
        } else if (arg == "-o" && i + 1 < argc) {
            outputDir = argv[++i];
            std::cout << "[STATUS] Output directory set to " << outputDir << "\n";
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Usage: " << argv[0] << " [-I bind_ip] [-d reconfig_delay] [-r number of rails] [-e emulation mode]\n"; return 1;
        }
    }
    std::error_code executableError;
    std::filesystem::path executablePath = std::filesystem::read_symlink("/proc/self/exe", executableError);
    std::string controllerDir = executableError ? "." : executablePath.parent_path().string();
    Controller server(bindIp, reconfigDelay, isEmulation, num_rails, topoIds, outputDir, controllerDir);
    return server.run();
    
}
