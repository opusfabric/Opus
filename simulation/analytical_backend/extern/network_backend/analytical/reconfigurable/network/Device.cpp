/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "reconfigurable/Device.h"
#include "reconfigurable/Chunk.h"
#include "reconfigurable/Link.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace NetworkAnalyticalReconfigurable;

std::function<void()> Device::increment_callback = []() { };
bool Device::drain_all_flow = true;

Device::Device(const DeviceId id) noexcept : device_id(id), topology_iteration(0), reconfiguring(false) {
    assert(id >= 0);
}

DeviceId Device::get_id() const noexcept {
    assert(device_id >= 0);
    return device_id;
}

std::shared_ptr<Link> Device::get_link(const DeviceId id) const noexcept {
    assert(id >= 0);
    assert(connected(id));
    return links.at(id);
}

struct LinkFreeCallbackArg {
    std::shared_ptr<Device> device_ptr;
    DeviceId link_id;
};

int Device::pending_chunks_count(const DeviceId id) const noexcept {
    assert(id >= 0);
    assert(connected(id));
    return static_cast<int>(pending_chunks.at(id).size());
}

void Device::link_become_free(DeviceId link_id) noexcept {
    
    // set link free
    links[link_id]->set_free();
    // std::cout << "Device " << device_id << ": link to " << link_id << " is free at time " << Link::get_current_time() << std::endl;


    // process pending chunks if one exist
    if(pending_chunks[link_id].empty() || pending_chunks[link_id].front()->get_topology_iteration() > topology_iteration) {
    // std::cout << "Device " << device_id << ": link to " << link_id << " is free but no pending chunks or chunk from future topology iteration. Pending queue size: " << pending_chunks[link_id].size() << std::endl;
        if(drain_all_flow){
            increment_callback();
        }

        return;
    }

    // printf("Pending chunk topology iteration: %d, current topology iteration: %d\n",
    //        pending_chunks[link_id].front()->get_topology_iteration(), topology_iteration);

    std::unique_ptr<Chunk> chunk = std::move(pending_chunks[link_id].front());
    pending_chunks[link_id].pop_front();

    if(links[link_id]->get_bandwidth() == Bandwidth(0)){
        // chunk->update_route(routes[link_id], chunk->get_topology_iteration()+1);
        // pending_chunks[link_id].push_back(std::move(chunk));
        std::cout << "UNROUTEABLE CHUNK, SAVING FOR NEXT TOPOLOGY Device " << device_id << ": link to " << link_id << " IS EMPTY" << pending_chunks[link_id].size() << std::endl;
        // link_become_free(link_id);
        // std::cout << "LINK FREE ERROR Device " << device_id << ": link to " << link_id << " IS EMPTY" << pending_chunks[link_id].size() << std::endl;
        // std::cout << "Chunk topology iteration: " << chunk->get_topology_iteration() << ", current topology iteration: " << topology_iteration << std::endl;
        assert(false);
        return;
    }
    auto next_link_free_time = links[link_id]->send(std::move(chunk));
    // schedule the next link free event
    // create a new callback argument for the next link free event
    LinkFreeCallbackArg* next_callback_arg = new LinkFreeCallbackArg{shared_from_this(), link_id};
    // get the next link free time
    
    // std::cout << "Device " << device_id << ": link to " << link_id << " becomes free at time and scheduled another chunk " << next_link_free_time << ", link pending chunk: " << pending_chunks[link_id].size() << std::endl;

    Link::schedule_event(next_link_free_time, link_become_free, next_callback_arg);
}

void Device::link_become_free(void* const arg) noexcept {
    assert(arg != nullptr);
    const auto* const callback_arg = static_cast<const LinkFreeCallbackArg*>(arg);
    assert(callback_arg->device_ptr != nullptr);
    assert(callback_arg->link_id >= 0);

    auto device = callback_arg->device_ptr;

    // invoke the link become free method on the device
    device->link_become_free(callback_arg->link_id);

    // clean up the callback argument
    
    delete callback_arg;
}

void Device::send(std::unique_ptr<Chunk> chunk) noexcept {
    // assert the validity of the chunk
    assert(chunk != nullptr);

    // assert this node is the current source of the chunk
    assert(chunk->current_device()->get_id() == device_id);

    // assert the chunk hasn't arrived its final destination yet
    assert(!chunk->arrived_dest());

    // Print out the route
    // for (const auto& [id, route] : routes) {
    //     std::cout << "Route to device " << id << ": ";
    //     for (const auto& hop : route) {
    //         std::cout << hop->get_id() << " ";
    //     }
    //     std::cout << std::endl;
    // }

    if (routes[chunk->next_device()->get_id()].empty()) {
        std::cout << "Device " << device_id << ": No route to device " << chunk->next_device()->get_id() << std::endl;
        std::cout << "Inserting Chunk to pending queue for device " << chunk->next_device()->get_id() << ", previous pending queue size: " << pending_chunks[chunk->next_device()->get_id()].size() << std::endl;
        pending_chunks[chunk->next_device()->get_id()].push_back(std::move(chunk));
        return;
    }

    chunk->update_route(routes[chunk->next_device()->get_id()], chunk->get_topology_iteration());
    //std::cout << "Device " << device_id << ": Sending chunk of size " << chunk->get_size() << " to device " << chunk->next_device()->get_id() << " via route: ";
    for (const auto& device : chunk->route) {
        std::cout << device->get_id() << " ";
    }
    std::cout << std::endl;

    // get next dest
    const auto next_dest = chunk->next_device();
    const auto next_dest_id = next_dest->get_id();

    // assert the next dest is connected to this node
    assert(connected(next_dest_id));

    auto link = links[next_dest_id];

    if (link->is_busy() || link->get_bandwidth() == Bandwidth(0) || chunk->get_topology_iteration() > topology_iteration) {
        // link is busy, add the chunk to pending chunks
        pending_chunks[next_dest_id].push_back(std::move(chunk));
        //std::cout << "Device " << device_id << ": link to " << next_dest_id << " is busy or reconfiguring, adding chunk to pending queue. Pending queue size: " << pending_chunks[next_dest_id].size() << std::endl;
        return;
    }

    // send the chunk to the next dest
    // delegate this task to the link

    if(links[next_dest_id]->get_bandwidth() == Bandwidth(0)){
        std::cout << "SEND ERROR Device " << device_id << ": link to " << next_dest_id << " IS EMPTY" << pending_chunks[next_dest_id].size() << std::endl;
        assert(false);
        pending_chunks[next_dest_id].push_back(std::move(chunk));
        return;
    }

    auto link_free_time = links[next_dest_id]->send(std::move(chunk));
    LinkFreeCallbackArg* args = new LinkFreeCallbackArg{shared_from_this(), next_dest_id};
    Link::schedule_event(link_free_time, link_become_free, args);
}

void Device::connect(const DeviceId id, const Bandwidth bandwidth, const Latency latency) noexcept {
    assert(id >= 0);
    assert(bandwidth >= 0);
    assert(latency >= 0);

    // assert there's no existing connection
    assert(!connected(id));

    // create link
    links[id] = std::make_shared<Link>(bandwidth, latency);
    pending_chunks[id] = std::list<std::unique_ptr<Chunk>>();
}

void Device::reconfigure(std::vector<Bandwidth> bandwidth, std::vector<Route> routes, std::vector<Latency> latency, Latency reconfig_time) noexcept {
    assert(bandwidth.size() == links.size());
    assert(latency.size() == links.size());

    topology_iteration++;

    for (const auto& [id, link] : links) {
        assert(id >= 0);

        if(id == device_id){
            continue;
        }

        assert(bandwidth[id] >= 0);
        assert(latency[id] >= 0);
        assert(connected(id));
 
        // update the route
        this->routes[id] = routes[id];
        // reconfigure the link
        //printf("Device %d: Reconfiguring link to %d, pending chunk size: %ld, new bandwidth: %f\n", device_id, id, pending_chunks[id].size(), bandwidth[id]);
        auto free_time = link->reconfigure(bandwidth[id], latency[id], reconfig_time);
        // create a callback argument for the link free event

        LinkFreeCallbackArg* args = new LinkFreeCallbackArg{shared_from_this(), id};
        // schedule the link free event
        Link::schedule_event(free_time, link_become_free, args);
    }

    // std::vector<std::unique_ptr<Chunk>> pending_chunks_copy;
    // // move pending chunks to a temporary vector
    // for (auto& [dest_id, queue] : pending_chunks) {
    //     while (!queue.empty()) {
    //         std::unique_ptr<Chunk> chunk = std::move(queue.front());
    //         queue.pop_front();
    //         if(chunk->get_topology_iteration() > topology_iteration) {
    //             queue.push_back(std::move(chunk));
    //         }
    //         else {
    //             pending_chunks_copy.push_back(std::move(chunk));
    //         }
    //     }
    // }

    // // re-assign routes for the pending chunks
    // for (auto& chunk : pending_chunks_copy) {
    //     assert(chunk != nullptr);
    //     assert(chunk->current_device()->get_id() == device_id);

    //     // re-assign the route
    //     chunk->update_route(routes[chunk->next_device()->get_id()], 0);

    //     // re-send the chunk
    //     send(std::move(chunk));
    // }
}

void Device::disconnect(const DeviceId id) noexcept {
    assert(id >= 0);

    // assert there's an existing connection
    assert(connected(id));

    // remove the link
    links.erase(id);
}

bool Device::connected(const DeviceId dest) const noexcept {
    assert(dest >= 0);

    // check whether the connection exists
    return links.find(dest) != links.end();
}
