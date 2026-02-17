#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <string>
#include "common/EventQueue.h"
#include "common/NetworkParser.h"
#include "reconfigurable/Chunk.h"
#include "reconfigurable/Helper.h"
#include "reconfigurable/Device.h"
#include "reconfigurable/Link.h"
#include "reconfigurable/TopologyManager.h"
#include <iostream>

using namespace std;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalReconfigurable;

static inline std::string trim(const std::string &s) {
    auto ws = " \t\r\n";
    size_t start = s.find_first_not_of(ws);
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(ws);
    return s.substr(start, end - start + 1);
}

void chunk_arrived_callback(void* const event_queue_ptr) {
    // typecast event_queue_ptr
    auto* const event_queue = static_cast<EventQueue*>(event_queue_ptr);

    // print chunk arrival time
    const auto current_time = event_queue->get_current_time();
    std::cout << "A chunk arrived at destination at time: " << current_time << " ns" << std::endl;
}

void simulate_trace(const std::string& path) noexcept {
    int npus_count = 0;
    int iters_count = 0;

    std::vector<std::vector<Bandwidth>> bw_matrix;
    int latency = 0;
    int reconfig_latency = 0;
    std::vector<std::tuple<int, int, int>> flows; // (src, dest, size)

    std::ifstream file(path);
    if (!file) {
        std::cerr << "[Error] (network/analytical/reconfigurable) " << "Failed to open trace file: " << path << std::endl;
        return;
    }

    const auto event_queue = std::make_shared<EventQueue>();
    Topology::set_event_queue(event_queue);
    std::unique_ptr<TopologyManager> tm;

    std::string line;
    bool is_bw_section = false;
    bool is_flow_section = false;

    std::vector<std::vector<Latency>> lt_matrix;


    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line.rfind("//", 0) == 0)
            continue;

        if (npus_count == 0 && isdigit(line[0])) {
            npus_count = stoi(line);
            printf("NPUs Count: %d\n", npus_count);
            tm = std::make_unique<TopologyManager>(npus_count, npus_count,event_queue.get());
            continue;
        }
        if (iters_count == 0 && isdigit(line[0])) {
            iters_count = stoi(line);
            printf("Iterations Count: %d\n", iters_count);
            bw_matrix.resize(npus_count, vector<Bandwidth>(npus_count));
            continue;
        }

        if (latency == 0 && isdigit(line[0])) {
            latency = stoi(line);
            lt_matrix.resize(npus_count, vector<Latency>(npus_count, Latency(latency)));
            continue;
        }

        if (reconfig_latency == 0 && isdigit(line[0])) {
            reconfig_latency = stoi(line);
            continue;
        }

        if (line == "BM" || line == "BW") {
            is_bw_section = true;
            bw_matrix.clear();
            is_flow_section = false;
            continue;
        } else if (line == "FLOW") {
            flows.clear();
            is_flow_section = true;
            is_bw_section = false;
            continue;
        }

        if (is_bw_section) {
            std::istringstream ss(line);
            std::vector<Bandwidth> row;
            int value;
            while (ss >> value) {
                row.push_back(Bandwidth(value));
            }
            bw_matrix.push_back(row);
            if(bw_matrix.size() == npus_count) {
                while(tm->is_reconfiguring() && !event_queue->finished()) {
                    event_queue->proceed();
                }
                if(tm->is_reconfiguring()) {
                    std::cerr << "[Error] (network/analytical/reconfigurable) " << "Internal Error: Reconfiguration incomplete." << std::endl;
                    return;
                }
                tm->reconfigure(bw_matrix, lt_matrix, Latency(reconfig_latency));
            }
        } 
        else if (is_flow_section) {
            auto p = line.find("->");
            line.replace(p, 2, " ");
            std::istringstream ss(line);
            int src, dest, size;
            if (ss >> src >> dest >> size) {
                flows.emplace_back(src, dest, size);
            }
            printf("Flow: %d -> %d, Size: %d\n", src, dest, size);
            tm->send(std::make_unique<Chunk>(size, tm->route(src, dest), chunk_arrived_callback, static_cast<void*>(event_queue.get()), -1));
        }
    }

    while (!event_queue->finished()) {
        event_queue->proceed();
    }

        // Print simulation result
    const auto finish_time = event_queue->get_current_time();
    std::cout << "Total NPUs Count: " << npus_count << std::endl;
    std::cout << "Simulation finished at time: " << finish_time << " ns" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <trace_file_path>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string trace_file_path = argv[1];
    simulate_trace(trace_file_path);

    return EXIT_SUCCESS;
}