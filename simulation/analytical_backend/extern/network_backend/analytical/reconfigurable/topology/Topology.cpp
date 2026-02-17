/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "reconfigurable/Topology.h"
#include "reconfigurable/Link.h"
#include <cassert>
#include <iostream>

using namespace NetworkAnalyticalReconfigurable;

void Topology::set_event_queue(std::shared_ptr<EventQueue> event_queue) noexcept {
    assert(event_queue != nullptr);

    // pass the given event_queue to Link
    Link::set_event_queue(std::move(event_queue));
}

int Topology::get_npus_count() const noexcept {
    return npus_count;
}

int Topology::get_devices_count() const noexcept {
    return devices_count;
}

Topology::Topology(int npus_count, int devices_count) noexcept {
    this->npus_count = npus_count;
    this->devices_count = devices_count;

    instantiate_devices();
    for (auto i = 0; i < devices_count; i++) {
        for (auto j = i+1; j < devices_count; j++) {
            // connect all devices to each other by default
            connect(i, j, Bandwidth(0), Latency(0), true);
        }
    }

    for (auto i = 0; i < devices_count; i++){
        connect(i, i, Bandwidth(0), Latency(0), false);
    }
};

std::shared_ptr<Device> Topology::get_device(const DeviceId deviceId) noexcept {
    assert(0 <= deviceId && deviceId < devices_count);
    return devices[deviceId];
}

void Topology::send(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);

    // get src npu node_id
    const auto src = chunk->current_device()->get_id();

    // assert src is valid
    assert(0 <= src && src < devices_count);

    // initiate transmission from src
    devices[src]->send(std::move(chunk));
}

void Topology::connect(const DeviceId src,
                       const DeviceId dest,
                       const Bandwidth bandwidth,
                       const Latency latency,
                       const bool bidirectional) noexcept {
    // assert the src and dest are valid
    assert(0 <= src && src < devices_count);
    assert(0 <= dest && dest < devices_count);

    // assert bandwidth and latency are valid
    assert(bandwidth >= 0);
    assert(latency >= 0);

    // std::cout << "Connecting device " << src << " to device " << dest
    //           << " with bandwidth: " << bandwidth
    //           << " and latency: " << latency << std::endl;

    // connect src -> dest
    devices[src]->connect(dest, bandwidth, latency);

    // if bidirectional, connect dest -> src
    if (bidirectional) {
        devices[dest]->connect(src, bandwidth, latency);
    }
}

void Topology::instantiate_devices() noexcept {
    // instantiate all devices
    for (auto i = 0; i < devices_count; i++) {
        devices.push_back(std::make_shared<Device>(i));
    }
}
