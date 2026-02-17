/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/EventQueue.h"
#include "common/Type.h"
#include "reconfigurable/Type.h"
#include <functional>
#include <memory>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalReconfigurable {

/**
 * Link models physical links between two devices.
 */
class Link {
  public:

    static int num_drained_links;

    static std::function<void()> increment_callback;

    /**
     * Set the event queue to be used by the link.
     *
     * @param event_queue_ptr pointer to the event queue
     */
    static void set_event_queue(std::shared_ptr<EventQueue> event_queue_ptr) noexcept;

    /**
     * Constructor.
     *
     * @param bandwidth bandwidth of the link
     * @param latency latency of the link
     */
    Link(Bandwidth bandwidth, Latency latency) noexcept;

    /**
     * Check if the link is busy.
     *
     * @return true if the link is busy, false otherwise
     */
    [[nodiscard]] bool is_busy() const noexcept;

    /**
     * Try to send a chunk through the link.
     * - If the link is free, service the chunk immediately.
     * - If the link is busy, add the chunk to the pending chunks list.
     *
     * @param chunk the chunk to be served by the link
     */
    unsigned long send(std::unique_ptr<Chunk> chunk) noexcept;

    /**
     * Set the link as busy.
     */
    void set_busy() noexcept;

    /**
     * Set the link as free.
     */
    void set_free() noexcept;

    /**
     * Reconfigure the link.
     */
    unsigned long reconfigure(Bandwidth bandwidth, Latency latency, Latency reconfig_time) noexcept;

    static int get_current_time() noexcept;

    /**
     * Get the bandwidth of the link in GB/s.
     *
     * @return bandwidth of the link
     */
    [[nodiscard]] Bandwidth get_bandwidth() const noexcept {
        return bandwidth;
    }

    ChunkSize get_total_transmitted_size() const noexcept {
        return total_transmitted_size;
    }

    static void schedule_event(EventTime event_time, Callback callback, void* const arg) noexcept;

  private:
    /// event queue Link uses to schedule events
    static std::shared_ptr<EventQueue> event_queue;

    /// bandwidth of the link in GB/s
    Bandwidth bandwidth;

    /// bandwidth of the link in B/ns, used in actual computation
    Bandwidth bandwidth_Bpns;

    /// latency of the link in ns
    Latency latency;

    /// Duration of the link
    EventTime duration;

    bool draining;

    EventTime pending_chunk_start_time;
    EventTime pending_chunk_end_time;
    ChunkSize pending_chunk_size;

    ChunkSize total_transmitted_size;

    /// flag to indicate if the link is busy
    bool busy;

    /**
     * Compute the serialization delay of a chunk on the link.
     * i.e., serialization delay = (chunk size) / (link bandwidth)
     *
     * @param chunk_size size of the target chunk
     * @return serialization delay of the chunk
     */
    [[nodiscard]] EventTime serialization_delay(ChunkSize chunk_size) const noexcept;

    /**
     * Compute the communication delay of a chunk.
     * i.e., communication delay = (link latency) + (serialization delay)
     *
     * @param chunk_size size of the target chunk
     * @return communication delay of the chunk
     */
    [[nodiscard]] EventTime communication_delay(ChunkSize chunk_size) const noexcept;

    /**
     * Schedule the transmission of a chunk.
     * - Set the link as busy.
     * - Link becomes free after the serialization delay.
     * - Chunk arrives next node after the communication delay.
     *
     * @param chunk chunk to be transmitted
     */
    unsigned long schedule_chunk_transmission(std::unique_ptr<Chunk> chunk) noexcept;
};

}  // namespace NetworkAnalyticalReconfigurable
