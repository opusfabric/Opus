
#include "reconfigurable/TopologyManager.h"
#include <cassert>
#include <algorithm>
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalReconfigurable;

TopologyManager::TopologyManager(int npus_count, int devices_count, EventQueue* event_queue, std::map<int, std::vector<std::vector<Bandwidth>>> circuit_schedules) noexcept {
    // Initialize the number of NPUs
    this->npus_count = npus_count;
    this->devices_count = devices_count;
    this->event_queue = event_queue;
    this->circuit_schedules = std::move(circuit_schedules);
    printf("Circuit schedules size: %zu\n", this->circuit_schedules.size());

    // Validate the counts
    assert(npus_count > 0);
    assert(devices_count > 0);
    assert(devices_count >= npus_count);

    reconfiguring = false;

    // Initialize the topology
    topology = std::make_shared<Topology>(npus_count, devices_count);

    Link::increment_callback = [this]() noexcept {
        // Increment the topology iteration
        this->increment_callback();
    };

    Device::increment_callback = [this]() noexcept {
        // Increment the topology iteration
        this->increment_callback();
    };

    // Initialize bandwidth and latency matrices
    bandwidths.resize(devices_count, std::vector<Bandwidth>(devices_count, Bandwidth(0)));
    latencies.resize(devices_count, std::vector<Latency>(devices_count, Latency(0)));

    topology_iteration = 0;

    inflight_coll = 0;
}

std::shared_ptr<Device> TopologyManager::get_device(const DeviceId deviceId) noexcept {
    // Validate the device ID
    assert(deviceId >= 0 && deviceId < devices_count);
    return topology->get_device(deviceId);
}

void TopologyManager::drain_network() noexcept {
    // Drain the network by iterating through all devices and their links
    Link::num_drained_links = 0;
    for (int i = 0; i < devices_count; ++i) {
        auto device = topology->get_device(i);
        device->draining = true;
        for(int j = 0; j < devices_count; ++j) {
            if (i != j) {
                auto link = device->get_link(j);
                if(!link->is_busy()) increment_callback();
                // TODO what if the link is busy, or the link does not exist
            }
        }
    }
}

bool TopologyManager::is_reconfiguring() const noexcept {
    return reconfiguring;
}

void TopologyManager::increment_callback() noexcept {
    if(!reconfiguring){
        Link::num_drained_links = 0;
        return;
    }

    // Increment the topology iteration
    Link::num_drained_links++;
    // printf("Link drained: %d/%d at %d\n", Link::num_drained_links, devices_count * (devices_count - 1), Link::get_current_time());

    if(Link::num_drained_links < devices_count * (devices_count - 1)) {
        // TODO: what if not all devices are connected to each other?
        return;
    }

    Link::num_drained_links = 0;
    reconfiguring = false;

    // All links have been drained, increment the topology iteration
    std::cout << "Drained Network, reconfiguring to TOPO ITERATION #" << topology_iteration << std::endl;
    for(auto row:bandwidths){
        for(auto bw: row){
            std::cout << bw << " ";
        }
        std::cout << std::endl;
    }


    for (int i = 0; i < devices_count; ++i) {
        auto device = topology->get_device(i);
        // std::vector<Route> routes;
        // Create a route for each device
        std::vector<Bandwidth> bw_device = bandwidths[i];
        std::cout << "BW Vector for device " << i << std::endl;
        for (auto bw: bw_device){
            std::cout << bw << " ";
        }
        std::cout << std::endl;

        device->reconfigure(bandwidths[i], precomputed_routes[i], latencies[i], reconfig_time);
    }
}

bool TopologyManager::reconfigure(std::vector<std::vector<Bandwidth>> bandwidths,
                               std::vector<std::vector<Latency>> latencies, Latency reconfig_time, int topo_id, int skip_inflight) noexcept {
    
    if (topo_id == cur_topo_id) {
        std::cout << "\033[1;31mTM: Already in the requested topology and reconfiguring, ignoring reconfiguration request to topo_id " << topo_id << "\033[0m" << std::endl;
        return true;
    }

    if (skip_inflight == 1 && inflight_coll > 0) {
        std::cout << "\033[1;31mTM: Force reconfig to topo_id " << topo_id << " despite inflight collectives: " << inflight_coll << "\033[0m" << std::endl;
    }

    if ((is_reconfiguring() || inflight_coll > 0) && skip_inflight == 0) {
        // TODO check condition
        // std::cout << "\033[1;31m\nTM: new topo_id " << topo_id << " cur_topo_id " << cur_topo_id << std::endl;
        // std::cout << "\033[1;31m\nTM: trying to reconfig, inflight coll: " << inflight_coll << ", is reconfiguring? " << is_reconfiguring() << ", is event queue finished? " << event_queue->finished() << "\033[0m" << std::endl;
        // // event_queue->proceed();
        return false;
    }

    printf("\n\033[1;31mTM: !!! TIME: %ld Reconfig to topo_id: %d, Devices count: %d, NPUs count: %d, inflight_coll %d\033[0m\n", event_queue->get_current_time(), topo_id, devices_count, npus_count, inflight_coll);
    printf("\033[1;31mTM: bandwidths size: %zu, latencies size: %zu\033[0m\n\n", bandwidths.size(), latencies.size());
    for(auto row:bandwidths){
        for(auto bw: row){
            std::cout << bw << " ";
        }
        std::cout << std::endl;
    }

    assert(bandwidths.size() == devices_count);
    assert(latencies.size() == devices_count);
    assert(!reconfiguring);

    for (const auto& row : bandwidths) {
        assert(row.size() == devices_count);
    }
    for (const auto& row : latencies) {
        assert(row.size() == devices_count);
    }

    // Update the bandwidth and latency matrices
    this->bandwidths = bandwidths;
    this->latencies = latencies;
    this->reconfig_time = reconfig_time;

    precomputeRoutes();

    // printf("TM: Reconfiguring topology with %d devices and %d NPUs and reconfig time %f.\n", devices_count, npus_count, reconfig_time);

    reconfiguring = true;
    this->cur_topo_id = topo_id;
    topology_iteration++;
    drain_network();
    return true;
}

bool TopologyManager::reconfigure(int topo_id, int skip_inflight) noexcept{
    auto it = circuit_schedules.find(topo_id);
    if (it != circuit_schedules.end()) {
        return reconfigure(it->second, latencies, reconfig_time, topo_id, skip_inflight);
    } else {
        printf("Topology ID %d not found in circuit schedules.\n", topo_id);
        exit(1);
    }
}

void TopologyManager::set_reconfig_latency(Latency latency) noexcept {
    this->reconfig_time = latency;
}

void TopologyManager::precomputeRoutes() noexcept {
    // TODO: add other routing algorithms 
    // Adjacency list
    std::vector<std::vector<int>> adj(devices_count);
    for (int i = 0; i < devices_count; ++i) {
        for (int j = 0; j < devices_count; ++j) {
            if (i != j && bandwidths[i][j] > 0) adj[i].push_back(j);
        }
    }

    for (auto& v : adj) {
        sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
    }


    precomputed_routes = std::vector<std::vector<Route>>(devices_count, std::vector<Route>(devices_count));

    // BFS
    const int INF = 1e9;
    std::vector<int> dist(devices_count), parent(devices_count);
    std::queue<int> q;

    for (int s = 0; s < devices_count; ++s) {
        // BFS init
        fill(dist.begin(), dist.end(), INF);
        fill(parent.begin(), parent.end(), -1);
        while (!q.empty()) q.pop();
        dist[s] = 0;
        q.push(s);

        // BFS
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                if (dist[v] == INF) {
                    dist[v] = dist[u] + 1;
                    parent[v] = u;
                    q.push(v);
                }
            }
        }

        // Reconstruct a path s -> t for all t
        for (int t = 0; t < devices_count; ++t) {
            if (s == t) {
                precomputed_routes[s][t] = {topology->get_device(s)};
            } else if (parent[t] == -1) {
                precomputed_routes[s][t] = {}; // Unreachable, stub route
            } else {
                Route path;
                for (int cur = t; cur != -1; cur = parent[cur]) path.push_back(topology->get_device(cur));
                reverse(path.begin(), path.end());
                precomputed_routes[s][t] = move(path);
            }
        }
    }
}

void TopologyManager::send(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);
    assert(chunk->current_device() != nullptr);
    // chunk->update_route(route(chunk->current_device()->get_id(), chunk->next_device()->get_id()), topology_iteration);

    // Get the source device ID
    DeviceId src = chunk->current_device()->get_id();
    assert(src >= 0 && src < devices_count);

    // if(chunk->get_topology_iteration() == -1){
    //     chunk->update_route(route(src, chunk->next_device()->get_id()), topology_iteration);
    // }

    // printf("TM: Sending chunk from %d to %d, in topo iter %d, route: ", chunk->current_device()->get_id(), chunk->next_device()->get_id(), chunk->get_topology_iteration());
    // for(auto device : chunk->route){
    //     printf("%d ", device->get_id());
    // }
    // printf("\n");

    // Send the chunk through the topology
    topology->send(std::move(chunk));
}

Route TopologyManager::route(DeviceId src, DeviceId dest) const noexcept {
    // Ensure src and dest are valid
    assert(src >= 0 && src < npus_count);
    assert(dest >= 0 && dest < npus_count);

    // Without any host forwarding.
    Route route;
    route.push_back(topology->get_device(src));

    // Create a route that includes the src and dest devices
    route.push_back(topology->get_device(dest));
    return route;
}

void TopologyManager::report_link_stats(std::ofstream& file) noexcept {
    file << "Link Statistics Report at time " << event_queue->get_current_time() << ":\n";
    for (int i = 0; i < devices_count; ++i) {
        auto device = topology->get_device(i);
        for (int j = 0; j < devices_count; ++j) {
            if (i != j) {
                auto link = device->get_link(j);
                if (link) {
                    file << "Link " << i << " -> " << j << ": Total Transmitted Size = " << link->get_total_transmitted_size() << " bytes\n";
                }
            }
        }
    }
}