
#include "reconfigurable/TopologyManager.h"
#include <cassert>
#include <algorithm>
#include <functional>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

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
    draining = false;

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
    if (!reconfiguring ||
        (routing_algorithm == RoutingAlgorithm::OPUS && !draining)) {
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
    if (routing_algorithm == RoutingAlgorithm::OPUS) {
        draining = false;
    } else {
        reconfiguring = false;
    }

    // All links have been drained, increment the topology iteration
    std::cout << "Drained Network, reconfiguring to TOPO ITERATION #" << topology_iteration << std::endl;

    for (int i = 0; i < devices_count; ++i) {
        auto device = topology->get_device(i);
        // std::vector<Route> routes;
        // Create a route for each device
        std::vector<Bandwidth> bw_device = bandwidths[i];
        // Device::reconfigure takes one Route per destination.  The chunk's
        // actual route is chosen at send time by TopologyManager::route(), so
        // we just pass the first ECMP candidate here (currently unused inside
        // Device, but kept for potential rerouting support).
        std::vector<Route> single_routes(devices_count);
        for (int j = 0; j < devices_count; ++j) {
            if (!ecmp_routes[i][j].empty())
                single_routes[j] = ecmp_routes[i][j][0];
        }
        device->reconfigure(bandwidths[i], single_routes, latencies[i], reconfig_time);
    }

    if (routing_algorithm == RoutingAlgorithm::OPUS) {
        if (reconfig_time == 0) {
            finish_reconfiguration();
        } else {
            event_queue->schedule_event(
                event_queue->get_current_time() + reconfig_time + 2,
                TopologyManager::finish_reconfiguration, this);
        }
    }
}

void TopologyManager::finish_reconfiguration(void* arg) noexcept {
    assert(arg != nullptr);
    static_cast<TopologyManager*>(arg)->finish_reconfiguration();
}

void TopologyManager::finish_reconfiguration() noexcept {
    reconfiguring = false;
    draining = false;
    Link::num_drained_links = 0;
}

bool TopologyManager::reconfigure(std::vector<std::vector<Bandwidth>> bandwidths,
                               std::vector<std::vector<Latency>> latencies, Latency reconfig_time, int topo_id, int skip_inflight) noexcept {
    
    if (topo_id == cur_topo_id) {
        if (routing_algorithm == RoutingAlgorithm::OPUS &&
            is_reconfiguring() && skip_inflight == 0) {
            return false;
        }
        return true;
    }

    if (routing_algorithm == RoutingAlgorithm::OPUS &&
        cur_topo_id != -1 && !is_reconfiguring() &&
        bandwidths == this->bandwidths &&
        latencies == this->latencies) {
        this->cur_topo_id = topo_id;
        this->reconfig_time = reconfig_time;
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
    draining = routing_algorithm == RoutingAlgorithm::OPUS;
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

void TopologyManager::set_routing_algorithm(RoutingAlgorithm algo) noexcept {
    routing_algorithm = algo;
}

// ---------------------------------------------------------------------------
// BFS routing: shortest hop count, single path per (src, dst).
// ---------------------------------------------------------------------------
static void bfs_routes(
        int N,
        const std::vector<std::vector<Bandwidth>>& bw,
        const std::shared_ptr<Topology>& topo,
        std::vector<std::vector<std::vector<Route>>>& out) {

    out.assign(N, std::vector<std::vector<Route>>(N));

    const int INF = 1e9;
    std::vector<int> dist(N), parent(N);
    std::queue<int> q;

    for (int s = 0; s < N; ++s) {
        fill(dist.begin(), dist.end(), INF);
        fill(parent.begin(), parent.end(), -1);
        dist[s] = 0;
        q.push(s);

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v = 0; v < N; ++v) {
                if (v != u && bw[u][v] > 0 && dist[v] == INF) {
                    dist[v] = dist[u] + 1;
                    parent[v] = u;
                    q.push(v);
                }
            }
        }

        for (int t = 0; t < N; ++t) {
            if (s == t) {
                out[s][t] = {{topo->get_device(s)}};
            } else if (parent[t] == -1) {
                out[s][t] = {};
            } else {
                Route path;
                for (int cur = t; cur != -1; cur = parent[cur])
                    path.push_back(topo->get_device(cur));
                std::reverse(path.begin(), path.end());
                out[s][t] = {std::move(path)};
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OPUS routing: hierarchical Dijkstra + ECMP.
//
// Edge weights:
//   intra-node edge (BW >= intra_threshold): cost 1
//   inter-node edge (0 < BW < intra_threshold): cost N+1
//
// The threshold is set to half of the maximum non-zero BW, so that the
// high-speed intra-node fabric is always preferred for lateral moves while
// the algorithm minimises the number of lower-speed scale-out hops.
//
// All paths achieving the same minimum Dijkstra cost are enumerated via
// predecessor-based DFS.  route() picks one uniformly at random (ECMP).
// ---------------------------------------------------------------------------
static void opus_routes(
        int N,
        const std::vector<std::vector<Bandwidth>>& bw,
        const std::shared_ptr<Topology>& topo,
        std::vector<std::vector<std::vector<Route>>>& out) {

    // Intra/inter threshold: half the peak BW
    float max_bw = 0.0f;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (bw[i][j] > max_bw) max_bw = bw[i][j];
    const float intra_threshold = max_bw * 0.5f;

    const int INTRA_W = 1;
    const int INTER_W = N + 1;   // one fewer scale-out hop always wins
    const int INF     = 1e9;

    out.assign(N, std::vector<std::vector<Route>>(N));

    using PQ = std::priority_queue<std::pair<int,int>,
                                   std::vector<std::pair<int,int>>,
                                   std::greater<std::pair<int,int>>>;

    std::vector<int>              dist(N);
    std::vector<std::vector<int>> preds(N);

    for (int s = 0; s < N; ++s) {
        fill(dist.begin(), dist.end(), INF);
        for (auto& p : preds) p.clear();
        dist[s] = 0;

        PQ pq;
        pq.push({0, s});

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            for (int v = 0; v < N; ++v) {
                if (v == u || bw[u][v] <= 0) continue;
                int w  = (bw[u][v] >= intra_threshold) ? INTRA_W : INTER_W;
                int nd = dist[u] + w;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    preds[v] = {u};
                    pq.push({nd, v});
                } else if (nd == dist[v]) {
                    preds[v].push_back(u);
                }
            }
        }

        // Enumerate a bounded set of shortest paths via predecessor DFS
        // (acyclic: dist strictly decreases toward s). Highly symmetric
        // scale-out topologies can have exponentially many equal-cost paths;
        // keeping a small ECMP candidate set preserves route diversity without
        // making precompute dominate the simulation.
        constexpr std::size_t kMaxEcmpRoutes = 1;
        std::vector<int> path_ids;
        path_ids.reserve(8);

        std::function<void(int, std::vector<Route>&)> dfs =
            [&](int cur, std::vector<Route>& paths) {
                if (paths.size() >= kMaxEcmpRoutes) return;
                path_ids.push_back(cur);
                if (cur == s) {
                    Route r;
                    for (int i = (int)path_ids.size() - 1; i >= 0; --i)
                        r.push_back(topo->get_device(path_ids[i]));
                    paths.push_back(std::move(r));
                } else {
                    for (int pred : preds[cur]) {
                        dfs(pred, paths);
                        if (paths.size() >= kMaxEcmpRoutes) break;
                    }
                }
                path_ids.pop_back();
            };

        for (int t = 0; t < N; ++t) {
            if (s == t) {
                out[s][t] = {{topo->get_device(s)}};
            } else if (dist[t] == INF) {
                out[s][t] = {};
            } else {
                std::vector<Route> paths;
                dfs(t, paths);
                out[s][t] = std::move(paths);
            }
        }
    }
}

void TopologyManager::precomputeRoutes() noexcept {
    if (routing_algorithm == RoutingAlgorithm::OPUS) {
        std::cout << "[TopologyManager] routing=opus (hierarchical Dijkstra + ECMP)" << std::endl;
        opus_routes(devices_count, bandwidths, topology, ecmp_routes);
    } else {
        std::cout << "[TopologyManager] routing=bfs (shortest-hop BFS)" << std::endl;
        bfs_routes(devices_count, bandwidths, topology, ecmp_routes);
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
    assert(src >= 0 && src < npus_count);
    assert(dest >= 0 && dest < npus_count);

    if (routing_algorithm != RoutingAlgorithm::OPUS) {
        Route direct;
        direct.push_back(topology->get_device(src));
        direct.push_back(topology->get_device(dest));
        return direct;
    }

    if (!ecmp_routes.empty() &&
        src  < (int)ecmp_routes.size() &&
        dest < (int)ecmp_routes[src].size() &&
        !ecmp_routes[src][dest].empty()) {

        const auto& candidates = ecmp_routes[src][dest];
        if (candidates.size() == 1) return candidates[0];

        // ECMP: pick uniformly at random among equal-cost paths.
        static thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
        return candidates[pick(rng)];
    }

    // Fallback before first reconfiguration: direct 2-hop route.
    Route r;
    r.push_back(topology->get_device(src));
    r.push_back(topology->get_device(dest));
    return r;
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