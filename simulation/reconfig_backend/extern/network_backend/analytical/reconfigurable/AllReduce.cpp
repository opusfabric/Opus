/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/EventQueue.h"
#include "common/NetworkParser.h"
#include "reconfigurable/Chunk.h"
#include "reconfigurable/Helper.h"
#include "reconfigurable/Device.h"
#include "reconfigurable/Link.h"
#include "reconfigurable/TopologyManager.h"
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalReconfigurable;

void chunk_arrived_callback(void* const event_queue_ptr) {
    // typecast event_queue_ptr
    auto* const event_queue = static_cast<EventQueue*>(event_queue_ptr);

    // print chunk arrival time
    const auto current_time = event_queue->get_current_time();
    std::cout << "A chunk arrived at destination at time: " << current_time << " ns" << std::endl;
}

int main() {
    // Instantiate shared resources
    const auto event_queue = std::make_shared<EventQueue>();
    Topology::set_event_queue(event_queue);

    // Parse network config and create topology
    // const auto network_parser = NetworkParser("../input/Spine.yml");
    // const auto topology = construct_topology(network_parser);
    // const auto npus_count = topology->get_npus_count();
    // const auto devices_count = topology->get_devices_count();

    const int npus_count = 4;  // Example NPU count
    const int devices_count = 4;  // Example devices count

    TopologyManager tm(npus_count, devices_count, event_queue.get());

    // event_queue->schedule_event(10000, reschedule_callback, static_cast<void*>(topology.get()));

    // message settings
    const auto chunk_size = 1'048'576;  // 1 MB
    
    tm.reconfigure(
        {{0, Bandwidth(200)}, {Bandwidth(200), 0}},  // bandwidths
        {{Latency(10), Latency(20)}, {Latency(20), Latency(10)}},            // latencies
        Latency(500)                                                         // reconfiguration time
    );

    // Run All-Gather
    for (int i = 0; i < npus_count; i++) {
        for (int j = 0; j < npus_count; j++) {
            if (i == j) {
                continue;
            }

            // create a chunk
            auto route = tm.route(i, j);
            auto* event_queue_ptr = static_cast<void*>(event_queue.get());
            auto chunk = std::make_unique<Chunk>(chunk_size, route, chunk_arrived_callback, event_queue_ptr,-1);

            // send a chunk
            tm.send(std::move(chunk));
        }
    }


    tm.reconfigure(
        {{0, Bandwidth(20)}, {Bandwidth(20), 0}},  // bandwidths
        {{Latency(10), Latency(20)}, {Latency(20), Latency(10)}},            // latencies
        Latency(500)                                                         // reconfiguration time
    );

    // event_queue->schedule_event(1000, reschedule_callback, static_cast<void*>(&tm));

    for (int i = 0; i < npus_count; i++) {
        for (int j = 0; j < npus_count; j++) {
            if (i == j) {
                continue;
            }

            // create a chunk
            auto route = tm.route(i, j);
            auto* event_queue_ptr = static_cast<void*>(event_queue.get());
            auto chunk = std::make_unique<Chunk>(chunk_size, route, chunk_arrived_callback, event_queue_ptr,-1);

            // send a chunk
            tm.send(std::move(chunk));
        }
    }

    // Run simulation
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    // std::cout << "Device 0 has " << count << " pending chunks to device 1." << std::endl;

    // Print simulation result
    const auto finish_time = event_queue->get_current_time();
    std::cout << "Total NPUs Count: " << npus_count << std::endl;
    std::cout << "Total devices Count: " << devices_count << std::endl;
    std::cout << "Simulation finished at time: " << finish_time << " ns" << std::endl;

    return 0;
}
