/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "reconfigurable/Type.h"
#include <map>
#include <memory>
#include <queue>
#include <functional>
#include <link.h>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalReconfigurable {

// class TopologyManager; // Forward declaration

/**
 * Device class represents a single device in the network.
 * Device is usually an NPU or a switch.
 */
class Device : public std::enable_shared_from_this<Device> {
  public:
    /**
     * Constructor.
     *
     * @param id id of the device
     */
    explicit Device(DeviceId id) noexcept;

    static std::function<void()> increment_callback; // Callback to be invoked when a link becomes free

    static bool drain_all_flow;
    /**
     * Get id of the device.
     *
     * @return id of the device
     */
    [[nodiscard]] DeviceId get_id() const noexcept;

        /**
     * Callback to be called when a link becomes free.
     *  - If the link has pending chunks, process the first one.
     *  - If the link has no pending chunks, set the link as free.
     *
     * @param link_ptr pointer to the link that becomes free
     */
    static void link_become_free(void* const arg) noexcept;

    void link_become_free(DeviceId link_id) noexcept;

    /**
     * Initiate a chunk transmission.
     * You must invoke this method on the source device of the chunk.
     *
     * @param chunk chunk to send
     */
    void send(std::unique_ptr<Chunk> chunk) noexcept;

    /**
     * Connect a device to another device.
     *
     * @param id id of the device to connect this device to
     * @param bandwidth bandwidth of the link
     * @param latency latency of the link
     */
    void connect(DeviceId id, Bandwidth bandwidth, Latency latency) noexcept;

    void reconfigure(std::vector<Bandwidth> bandwidths,
                     std::vector<Route> routes,
                     std::vector<Latency> latencies,
                     Latency reconfigTime) noexcept;

    /**
     * Disconnect a device from another device.
     *
     * @param id id of the device to disconnect this device from
     */
    void disconnect(DeviceId id) noexcept;

    int pending_chunks_count(DeviceId id) const noexcept;

    std::shared_ptr<Link> get_link(DeviceId id) const noexcept;

    bool draining;

    bool reconfiguring;

  private:
    /// device Id
    DeviceId device_id;

    int topology_iteration;

    /// links to other nodes
    /// map[dest node node_id] -> link

    std::map<DeviceId, std::shared_ptr<Link>> links;
    std::map<DeviceId, std::list<std::unique_ptr<Chunk>>> pending_chunks;
    std::map<DeviceId, Route> routes;

    /**
     * Check if this device is connected to another device.
     *
     * @param dest id of the device to check te connectivity
     * @return true if connected to the given device, false otherwise
     */
    [[nodiscard]] bool connected(DeviceId dest) const noexcept;
};

}  // namespace NetworkAnalyticalReconfigurable
