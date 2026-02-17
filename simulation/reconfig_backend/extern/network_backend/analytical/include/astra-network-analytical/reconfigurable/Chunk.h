/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "reconfigurable/Type.h"
#include <memory>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalReconfigurable {

/**
 * Chunk class represents a chunk.
 * Chunk is a basic unit of transmission.
 */
class Chunk {
  public:
    /**
     * Callback to be invoked when a chunk arrives at the next device.
     *   - if the chunk arrived at its destination, the final callback is invoked
     *   - if not, the chunk is sent to the next device as designated by the route
     *
     * @param chunk_ptr: pointer to the chunk that's arrived at the next device
     */
    static void chunk_arrived_next_device(void* chunk_ptr) noexcept;


    /**
     * Constructor.
     *
     * @param chunk_size: size of the chunk
     * @param route: route of the chunk from its source to destination
     * @param callback: callback to be invoked when the chunk arrives destination
     * @param callback_arg: argument of the callback
     */
    Chunk(ChunkSize chunk_size, Route route, Callback callback, CallbackArg callback_arg, int topology_iteration) noexcept;

    Chunk(ChunkSize chunk_size, Route route, Callback callback, CallbackArg callback_arg) noexcept
        : Chunk(chunk_size, std::move(route), callback, callback_arg, -1) {}

    bool no_route() noexcept {
        return route.size() <= 1;
    }

    void update_route(Route new_route, int topology_iteration) noexcept {
        route = std::move(new_route);
        this->topology_iteration = topology_iteration;
    }

    int get_topology_iteration() const noexcept {
        return topology_iteration;
    }

    /**
     * Get the current sitting device of the chunk
     *
     * @return current device of the chunk
     */
    [[nodiscard]] std::shared_ptr<Device> current_device() const noexcept;

    /**
     * Get the next destined device of the chunk
     *
     * @return next device of the chunk
     */
    [[nodiscard]] std::shared_ptr<Device> next_device() const noexcept;

    /**
     * Mark the chunk arrived at its next device
     * i.e., drop the current device from the route
     */
    void mark_arrived_next_device() noexcept;

    /**
     * Check if the chunk arrived at its destination
     * i.e., if the route length is 1 (only destination device left)
     *
     * @return true if the chunk arrived at its destination, false otherwise
     */
    [[nodiscard]] bool arrived_dest() const noexcept;

    /**
     * Get the size of the chunk
     *
     * @return size of the chunk
     */
    [[nodiscard]] ChunkSize get_size() const noexcept;

    /**
     * Invoke the registered callback
     * i.e., this method should be called when the chunk arrives its destination.
     */
    void invoke_callback() noexcept;

    void static add_on_route_chunk() noexcept {
        on_route_chunks++;
    }

    [[nodiscard]] static int get_on_route_chunks() noexcept {
        return on_route_chunks;
    }

    Route route;

  private:
    static int on_route_chunks;

    /// size of the chunk
    ChunkSize chunk_size;

    /// route of the chunk to its destination.
    /// Route has the structure of [current device, next device, ..., dest device]
    /// e.g., if a chunk starts from device 5, then reaches destination 3,
    /// the route would be e.g., [5, 1, 6, 2, 3]

    DeviceId src_id;

    DeviceId dest_id;

    /// callback to be invoked when the chunk arrives at its destination
    Callback callback;

    /// argument of the callback
    CallbackArg callback_arg;

    int topology_iteration;
};

}  // namespace NetworkAnalyticalReconfigurable
