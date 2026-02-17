/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/EventQueue.h"
#include "reconfigurable/Chunk.h"
#include "reconfigurable/Device.h"
#include "reconfigurable/Topology.h"
#include "reconfigurable/Type.h"
#include "reconfigurable/Link.h"
#include <memory>
#include <vector>
#include <fstream>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalReconfigurable {

/**
 * Topology abstracts a network topology.
 */
class TopologyManager {
  public:
    /**
     * Constructor.
     */
    TopologyManager(int npus_count, int devices_count, EventQueue* event_queue, std::map<int, std::vector<std::vector<Bandwidth>>> circuit_schedules) noexcept;

    std::shared_ptr<Device> get_device(const DeviceId deviceId) noexcept;

    /**
     * Reconfigure the topology with new bandwidths and latencies.
     */
    bool reconfigure(std::vector<std::vector<Bandwidth>> bandwidths,
                     std::vector<std::vector<Latency>> latencies, Latency reconfig_time, int topo_id=0, int skip_inflight=0) noexcept;

    bool reconfigure(int topo_id, int skip_inflight=0) noexcept;

    void set_reconfig_latency(Latency latency) noexcept;

    void precomputeRoutes() noexcept;

    void precomputeSingleRoute(DeviceId src, DeviceId dst) noexcept;

    void drain_network() noexcept;

    void increment_callback() noexcept;

    void set_cur_topo_id(int topo_id) noexcept {
        cur_topo_id = topo_id;
        return;
    };

    void report_link_stats(std::ofstream& file) noexcept;
    
    int inflight_coll;

    /**
     * Construct the route from src to dest.
     * Route is a list of devices (pointers) that the chunk should traverse,
     * including the src and dest devices themselves.
     *
     * e.g., route(0, 3) = [0, 5, 7, 2, 3]
     *
     * @param src src NPU id
     * @param dest dest NPU id
     *
     * @return route from src NPU to dest NPU
     */
    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept;

    /**
     * Initiate a transmission of a chunk.
     *
     * @param chunk chunk to be transmitted
     */
    void send(std::unique_ptr<Chunk> chunk) noexcept;

    /**
     * Get the number of NPUs in the topology.
     * NPU excludes non-NPU devices such as switches.
     *
     * @return number of NPUs in the topology
     */
    [[nodiscard]] int get_npus_count() const noexcept;

    /**
     * Get the number of devices in the topology.
     * Device includes non-NPU devices such as switches.
     *
     * @return number of devices in the topology
     */
    [[nodiscard]] int get_devices_count() const noexcept;

    bool is_reconfiguring() const noexcept;

  protected:
    /// number of total devices in the topology
    /// device includes non-NPU devices such as switches
    int devices_count;

    /// event queue to be used by the topology
    EventQueue* event_queue;

    /// number of NPUs in the topology
    /// NPU excludes non-NPU devices such as switches
    int npus_count;

    int topology_iteration;

    int cur_topo_id = -1;

    bool reconfiguring;

    Latency reconfig_time;

    /// holds the entire topology
    std::shared_ptr<Topology> topology;
    
    /// bandwidth matrix
    std::vector<std::vector<Bandwidth>> bandwidths;

    /// latency matrix
    std::vector<std::vector<Latency>> latencies;

    std::vector<std::vector<Route>> precomputed_routes;

    std::map<int, std::vector<std::vector<Bandwidth>>> circuit_schedules;
};

}  // namespace NetworkAnalyticalReconfigurable
