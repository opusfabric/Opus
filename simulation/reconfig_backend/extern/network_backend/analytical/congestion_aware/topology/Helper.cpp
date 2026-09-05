/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Helper.h"
#include "congestion_aware/FullyConnected.h"
#include "congestion_aware/Ring.h"
#include "congestion_aware/Switch.h"
#include <cassert>
#include <cstdlib>
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

class PerRailTopology final : public Topology {
  public:
    PerRailTopology(const int scale_up_npus,
                    const int scale_out_nodes,
                    const Bandwidth scale_up_bandwidth,
                    const Bandwidth scale_out_bandwidth,
                    const Bandwidth cross_rail_bandwidth,
                    const Latency scale_up_latency,
                    const Latency scale_out_latency,
                    const Latency cross_rail_latency,
                    const bool fully_connected_scale_up,
                    const bool fully_connected_scale_out,
                    const bool cross_rail_switch) noexcept
        : scale_up_npus(scale_up_npus),
          scale_out_nodes(scale_out_nodes),
          fully_connected_scale_up(fully_connected_scale_up),
          fully_connected_scale_out(fully_connected_scale_out),
          cross_rail_switch(cross_rail_switch),
          local_switch_base(-1),
          rail_switch_base(-1),
          cross_rail_switch_device_id(-1) {
        assert(scale_up_npus > 1);
        assert(scale_out_nodes > 1);
        assert(scale_up_bandwidth > 0);
        assert(scale_out_bandwidth > 0);
        assert(!cross_rail_switch || cross_rail_bandwidth > 0);
        assert(!cross_rail_switch || !fully_connected_scale_out);
        assert(scale_up_latency >= 0);
        assert(scale_out_latency >= 0);
        assert(!cross_rail_switch || cross_rail_latency >= 0);

        dims_count = cross_rail_switch ? 3 : 2;
        npus_count = scale_up_npus * scale_out_nodes;
        local_switch_base = fully_connected_scale_up ? -1 : npus_count;
        rail_switch_base = fully_connected_scale_out
            ? -1
            : npus_count + (fully_connected_scale_up ? 0 : scale_out_nodes);
        cross_rail_switch_device_id = cross_rail_switch ? rail_switch_base + scale_up_npus : -1;
        devices_count = npus_count +
                        (fully_connected_scale_up ? 0 : scale_out_nodes) +
                        (fully_connected_scale_out ? 0 : scale_up_npus) +
                        (cross_rail_switch ? 1 : 0);
        npus_count_per_dim = cross_rail_switch
            ? std::vector<int>{scale_up_npus, scale_out_nodes, scale_up_npus}
            : std::vector<int>{scale_up_npus, scale_out_nodes};
        bandwidth_per_dim = cross_rail_switch
            ? std::vector<Bandwidth>{scale_up_bandwidth, scale_out_bandwidth, cross_rail_bandwidth}
            : std::vector<Bandwidth>{scale_up_bandwidth, scale_out_bandwidth};

        instantiate_devices();

        if (fully_connected_scale_up) {
            for (int node = 0; node < scale_out_nodes; node++) {
                for (int src_rail = 0; src_rail < scale_up_npus; src_rail++) {
                    for (int dest_rail = 0; dest_rail < scale_up_npus; dest_rail++) {
                        if (src_rail != dest_rail) {
                            connect(npu_id(node, src_rail),
                                    npu_id(node, dest_rail),
                                    scale_up_bandwidth,
                                    scale_up_latency,
                                    false);
                        }
                    }
                }
            }
        } else {
            for (int node = 0; node < scale_out_nodes; node++) {
                const auto local_switch = local_switch_id(node);
                for (int rail = 0; rail < scale_up_npus; rail++) {
                    const auto npu = npu_id(node, rail);
                    connect(npu, local_switch, scale_up_bandwidth, scale_up_latency, true);
                }
            }
        }

        if (fully_connected_scale_out) {
            for (int rail = 0; rail < scale_up_npus; rail++) {
                for (int src_node = 0; src_node < scale_out_nodes; src_node++) {
                    for (int dest_node = 0; dest_node < scale_out_nodes; dest_node++) {
                        if (src_node != dest_node) {
                            connect(npu_id(src_node, rail),
                                    npu_id(dest_node, rail),
                                    scale_out_bandwidth,
                                    scale_out_latency,
                                    false);
                        }
                    }
                }
            }
        } else {
            for (int rail = 0; rail < scale_up_npus; rail++) {
                const auto rail_switch = rail_switch_id(rail);
                for (int node = 0; node < scale_out_nodes; node++) {
                    const auto endpoint = fully_connected_scale_up ? npu_id(node, rail) : local_switch_id(node);
                    connect(endpoint, rail_switch, scale_out_bandwidth, scale_out_latency, true);
                }
                if (cross_rail_switch) {
                    connect(rail_switch, cross_rail_switch_device_id, cross_rail_bandwidth, cross_rail_latency, true);
                }
            }
        }
    }

    [[nodiscard]] Route route(const DeviceId src, const DeviceId dest) const noexcept override {
        assert(0 <= src && src < npus_count);
        assert(0 <= dest && dest < npus_count);

        auto route = Route();
        route.push_back(devices[src]);

        if (src == dest) {
            return route;
        }

        const auto src_node = src / scale_up_npus;
        const auto src_rail = src % scale_up_npus;
        const auto dest_node = dest / scale_up_npus;
        const auto dest_rail = dest % scale_up_npus;

        if (src_node == dest_node) {
            if (!fully_connected_scale_up) {
                route.push_back(devices[local_switch_id(src_node)]);
            }
            route.push_back(devices[dest]);
            return route;
        }

        if (fully_connected_scale_out) {
            if (src_rail != dest_rail) {
                if (!fully_connected_scale_up) {
                    route.push_back(devices[local_switch_id(src_node)]);
                }
                route.push_back(devices[npu_id(src_node, dest_rail)]);
            }
            route.push_back(devices[dest]);
            return route;
        }

        if (cross_rail_switch && src_rail != dest_rail) {
            if (!fully_connected_scale_up) {
                route.push_back(devices[local_switch_id(src_node)]);
            }
            route.push_back(devices[rail_switch_id(src_rail)]);
            route.push_back(devices[cross_rail_switch_device_id]);
            route.push_back(devices[rail_switch_id(dest_rail)]);
            if (!fully_connected_scale_up) {
                route.push_back(devices[local_switch_id(dest_node)]);
            }
            route.push_back(devices[dest]);
            return route;
        }

        if (fully_connected_scale_up) {
            if (src_rail != dest_rail) {
                route.push_back(devices[npu_id(src_node, dest_rail)]);
            }
            route.push_back(devices[rail_switch_id(dest_rail)]);
            route.push_back(devices[dest]);
            return route;
        }

        route.push_back(devices[local_switch_id(src_node)]);
        route.push_back(devices[rail_switch_id(src_rail)]);
        route.push_back(devices[local_switch_id(dest_node)]);
        route.push_back(devices[dest]);

        return route;
    }

  private:
    [[nodiscard]] DeviceId npu_id(const int node, const int rail) const noexcept {
        assert(0 <= node && node < scale_out_nodes);
        assert(0 <= rail && rail < scale_up_npus);
        return node * scale_up_npus + rail;
    }

    [[nodiscard]] DeviceId local_switch_id(const int node) const noexcept {
        assert(!fully_connected_scale_up);
        assert(0 <= node && node < scale_out_nodes);
        return local_switch_base + node;
    }

    [[nodiscard]] DeviceId rail_switch_id(const int rail) const noexcept {
        assert(0 <= rail && rail < scale_up_npus);
        return rail_switch_base + rail;
    }

    int scale_up_npus;
    int scale_out_nodes;
    bool fully_connected_scale_up;
    bool fully_connected_scale_out;
    bool cross_rail_switch;
    DeviceId local_switch_base;
    DeviceId rail_switch_base;
    DeviceId cross_rail_switch_device_id;
};

std::shared_ptr<Topology> construct_basic_topology(
    const TopologyBuildingBlock topology_type,
    const int npus_count,
    const Bandwidth bandwidth,
    const Latency latency) noexcept {
    switch (topology_type) {
    case TopologyBuildingBlock::Ring:
        return std::make_shared<Ring>(npus_count, bandwidth, latency);
    case TopologyBuildingBlock::Switch:
        return std::make_shared<Switch>(npus_count, bandwidth, latency);
    case TopologyBuildingBlock::FullyConnected:
        return std::make_shared<FullyConnected>(npus_count, bandwidth, latency);
    default:
        std::cerr << "[Error] (network/analytical/congestion_aware) "
                  << "not supported basic-topology" << std::endl;
        std::exit(-1);
    }
}

}  // namespace

std::shared_ptr<Topology> NetworkAnalyticalCongestionAware::construct_topology(
    const NetworkParser& network_parser) noexcept {
    const auto dims_count = network_parser.get_dims_count();
    const auto topologies_per_dim = network_parser.get_topologies_per_dim();
    const auto npus_counts_per_dim = network_parser.get_npus_counts_per_dim();
    const auto bandwidths_per_dim = network_parser.get_bandwidths_per_dim();
    const auto latencies_per_dim = network_parser.get_latencies_per_dim();

    if (dims_count == 1) {
        return construct_basic_topology(
            topologies_per_dim[0],
            npus_counts_per_dim[0],
            bandwidths_per_dim[0],
            latencies_per_dim[0]);
    }

    const bool supported_per_rail =
        (dims_count == 2 || dims_count == 3) &&
        (topologies_per_dim[0] == TopologyBuildingBlock::Switch ||
         topologies_per_dim[0] == TopologyBuildingBlock::FullyConnected) &&
        (topologies_per_dim[1] == TopologyBuildingBlock::Switch ||
         (dims_count == 2 && topologies_per_dim[1] == TopologyBuildingBlock::FullyConnected)) &&
        (dims_count == 2 || topologies_per_dim[2] == TopologyBuildingBlock::Switch);

    if (supported_per_rail) {
        if (dims_count == 3 && npus_counts_per_dim[2] != npus_counts_per_dim[0]) {
            std::cerr << "[Error] (network/analytical/congestion_aware) "
                      << "cross-rail switch dimension must match the rail count"
                      << std::endl;
            std::exit(-1);
        }

        const bool fully_connected_scale_out = topologies_per_dim[1] == TopologyBuildingBlock::FullyConnected;
        const bool cross_rail_switch = dims_count == 3;
        return std::make_shared<PerRailTopology>(
            npus_counts_per_dim[0],
            npus_counts_per_dim[1],
            bandwidths_per_dim[0],
            bandwidths_per_dim[1],
            cross_rail_switch ? bandwidths_per_dim[2] : 0,
            latencies_per_dim[0],
            latencies_per_dim[1],
            cross_rail_switch ? latencies_per_dim[2] : 0,
            topologies_per_dim[0] == TopologyBuildingBlock::FullyConnected,
            fully_connected_scale_out,
            cross_rail_switch);
    }

    std::cerr << "[Error] (network/analytical/congestion_aware) "
              << "only support 1-dim topology or 2/3-dim per-rail Switch/FullyConnected topology"
              << std::endl;
    std::exit(-1);
}
