/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "reconfigurable/Link.h"
#include "common/NetworkFunction.h"
#include "reconfigurable/Chunk.h"
#include "reconfigurable/Device.h"
#include <cassert>
#include <algorithm>
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalReconfigurable;

// declaring static event_queue
std::shared_ptr<EventQueue> Link::event_queue;


int NetworkAnalyticalReconfigurable::Link::num_drained_links = 0;
std::function<void()> Link::increment_callback = []() { };

void Link::set_event_queue(std::shared_ptr<EventQueue> event_queue_ptr) noexcept {
    assert(event_queue_ptr != nullptr);

    // set the event queue
    Link::event_queue = std::move(event_queue_ptr);
}

int Link::get_current_time() noexcept {
    assert(event_queue != nullptr);

    // return current time of the event queue
    return Link::event_queue->get_current_time();
}

void Link::schedule_event(EventTime event_time, Callback callback, void* const arg) noexcept {
    assert(event_queue != nullptr);
    assert(callback != nullptr);

    // schedule the event in the event queue
    Link::event_queue->schedule_event(event_time, callback, arg);
}

Link::Link(const Bandwidth bandwidth, const Latency latency) noexcept
    : bandwidth(bandwidth),
      latency(latency),
      busy(false) {
    assert(bandwidth >= 0);
    assert(latency >= 0);

    // convert bandwidth from GB/s to B/ns
    bandwidth_Bpns = bw_GBps_to_Bpns(bandwidth);
    pending_chunk_end_time = 0;
    total_transmitted_size = 0;
}

bool Link::is_busy() const noexcept {
    // return busy state
    return busy;
}

unsigned long Link::send(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);
    assert(!busy);
    assert(bandwidth > 0);
    
    return schedule_chunk_transmission(std::move(chunk));
}

void Link::set_busy() noexcept {
    // set busy to true
    busy = true;
}

void Link::set_free() noexcept {
    // set busy to false
    busy = false;
}

EventTime Link::serialization_delay(const ChunkSize chunk_size) const noexcept {
    assert(chunk_size > 0);

    // calculate serialization delay
    const auto delay = static_cast<Bandwidth>(chunk_size) / bandwidth_Bpns;

    // return serialization delay in EventTime type
    return static_cast<EventTime>(delay);
}

EventTime Link::communication_delay(const ChunkSize chunk_size) const noexcept {
    assert(chunk_size > 0);

    // calculate communication delay
    const auto delay = latency + (static_cast<Bandwidth>(chunk_size) / bandwidth_Bpns);

    // return communication delay in EventTime type
    return static_cast<EventTime>(delay);
}

unsigned long Link::schedule_chunk_transmission(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);

    // link should be free
    assert(!busy);

    // set link busy
    set_busy();

    // get metadata
    const auto chunk_size = chunk->get_size();
    const auto current_time = Link::event_queue->get_current_time();

    // schedule chunk arrival event
    const auto communication_time = communication_delay(chunk_size);
    const auto chunk_arrival_time = current_time + communication_time;
    auto* const chunk_ptr = static_cast<void*>(chunk.release());
    Link::event_queue->schedule_event(chunk_arrival_time, Chunk::chunk_arrived_next_device, chunk_ptr);
    
    // schedule link free time
    const auto serialization_time = serialization_delay(chunk_size);
    const auto link_free_time = current_time + serialization_time + 2 * latency;
    pending_chunk_end_time = link_free_time;
    total_transmitted_size += chunk_size;
    return link_free_time;
}

unsigned long Link::reconfigure(Bandwidth bandwidth, Latency latency, Latency reconfig_time) noexcept{
    if (bandwidth == this->bandwidth && latency == this->latency) {
        // std::cout << "No reconfiguration needed" << std::endl;
        // if(busy) {
        //     std::cout << "Link is busy during no-op reconfiguration" << std::endl;
        //     std::cout << "Current time: " << Link::event_queue->get_current_time() << " ns" << std::endl;
        //     std::cout << "Chunk end time: " << pending_chunk_end_time << " ns" << std::endl;
        // }
        set_busy();
        return Link::event_queue->get_current_time() + 1;
    }

    // assert(!busy);
    const auto current_time = Link::event_queue->get_current_time();
    set_busy();

    //printf("Reconfiguring link from bandwidth %.2f GB/s to %.2f GB/s and latency %.2f ns to %.2f ns at time %lu ns\n",
    //       this->bandwidth, bandwidth, this->latency, latency, current_time);

    this->bandwidth = bandwidth;
    this->latency = latency;
    this->bandwidth_Bpns = bw_GBps_to_Bpns(bandwidth);

    // printf("Link bandwidth updated to %.2f B/ns and latency updated to %.2f ns\n",
    //        this->bandwidth_Bpns, this->latency);

    const auto reconfig_completion_time = current_time + reconfig_time + 1;

    auto* const link_ptr = static_cast<void*>(this);
    return reconfig_completion_time;
}