/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "reconfigurable/ReconfigurableNetworkApi.hh"
#include <astra-network-analytical/reconfigurable/Chunk.h>
#include <cassert>
#include <iostream>
#include <unordered_set>

using namespace AstraSim;
using namespace AstraSimAnalyticalReconfigurable;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalReconfigurable;

std::shared_ptr<TopologyManager> ReconfigurableNetworkApi::tm;
std::map<int, std::unordered_set<std::string>> ReconfigurableNetworkApi::on_going_comms;

void ReconfigurableNetworkApi::set_topology(
    std::shared_ptr<TopologyManager> tm_ptr) noexcept {
    assert(tm_ptr != nullptr);

    // move topology
    ReconfigurableNetworkApi::tm = std::move(tm_ptr);
    ReconfigurableNetworkApi::on_going_comms.clear();
}

ReconfigurableNetworkApi::ReconfigurableNetworkApi(const int rank) noexcept
    : CommonNetworkApi(rank) {
    assert(rank >= 0);
}

bool ReconfigurableNetworkApi::sim_reconfig(int topo_id, int skip_inflight) {
    return tm->reconfigure(topo_id, skip_inflight);
}

void ReconfigurableNetworkApi::increment_inflight_coll(int rank, std::string name) {
    if(on_going_comms.find(rank) == on_going_comms.end()) {
        on_going_comms[rank] = std::unordered_set<std::string>();
    }
    
    on_going_comms[rank].insert(name);
    tm->inflight_coll++;
}

void ReconfigurableNetworkApi::decrement_inflight_coll(int rank, int node_id) {
    if (on_going_comms.find(rank) == on_going_comms.end()) {
        std::cerr << "Error: Rank " << rank << " not initialized" << std::endl;
        assert(false);
    }
    if (node_id < 0) {
        for (const auto& name : on_going_comms[rank]) {
            if (name.find("COLL") != std::string::npos) {
                on_going_comms[rank].erase(name);
                break;
            }
        }
    }
    else { 
        for (const auto& name : on_going_comms[rank]) {
            if (name.find(std::to_string(node_id)) != std::string::npos) {
                on_going_comms[rank].erase(name);
            break;
            }

        }
    }
    if(tm->inflight_coll == 0) {
        std::cerr << "Error: inflight_coll is already zero!" << std::endl;
        assert(false);
    }
    tm->inflight_coll--;
}

void ReconfigurableNetworkApi::print_on_going_comms() {
    std::cout << "On going comms: " << std::endl;
    for (const auto& [rank, set] : on_going_comms) {
        if (set.empty()) continue;
        std::cout << "Rank " << rank;
        for (const auto& name : set) {
            std::cout << " " << name << ", ";
        }
        std::cout << std::endl;
    }
}

int ReconfigurableNetworkApi::get_inflight_coll() {
    return tm->inflight_coll;
}

int ReconfigurableNetworkApi::sim_send(void* const buffer,
                                        const uint64_t count,
                                        const int type,
                                        const int dst,
                                        const int tag,
                                        sim_request* const request,
                                        void (*msg_handler)(void*),
                                        void* const fun_arg) {
    // query chunk id
    const auto src = sim_comm_get_rank();
    const auto chunk_id =
        ReconfigurableNetworkApi::chunk_id_generator.create_send_chunk_id(
            tag, src, dst, count);

    // search tracker
    const auto entry =
        callback_tracker.search_entry(tag, src, dst, count, chunk_id);
    if (entry.has_value()) {
        // recv operation already issued.
        // register send callback
        entry.value()->register_send_callback(msg_handler, fun_arg);
    } else {
        // recv operation not issued yet
        // create new entry and insert callback
        auto* const new_entry =
            callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_send_callback(msg_handler, fun_arg);
    }

    // create chunk
    auto chunk_arrival_arg = std::tuple(tag, src, dst, count, chunk_id);
    auto arg = std::make_unique<decltype(chunk_arrival_arg)>(chunk_arrival_arg);
    const auto arg_ptr = static_cast<void*>(arg.release());
    const auto route = tm->route(src, dst);
    auto chunk = std::make_unique<Chunk>(
        count, route, ReconfigurableNetworkApi::process_chunk_arrival,
        arg_ptr);

    // initiate transmission from src -> dst.
    tm->send(std::move(chunk));

    // return
    return 0;
}
