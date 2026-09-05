/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/AllToAllv.hh"

#include <algorithm>
#include <cmath>

#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/Sys.hh"

using namespace AstraSim;

AllToAllv::AllToAllv(ComType type,
                     int window,
                     int id,
                     RingTopology* allToAllTopology,
                     uint64_t data_size,
                     RingTopology::Direction direction,
                     InjectionPolicy injection_policy,
                     AllToAllVLoadMapPtr all2allv_loads,
                     uint64_t all2allv_total_size,
                     uint64_t all2allv_chunk_size)
    : AllToAll(type, window, id, allToAllTopology, data_size, direction,
               injection_policy) {
    this->name = Name::AllToAllv;
    this->all2allv_loads = all2allv_loads;
    this->all2allv_total_size = all2allv_total_size;
    this->all2allv_chunk_size = all2allv_chunk_size;
}

uint64_t AllToAllv::pair_msg_size(int src, int dst) const {
    if (comType != ComType::All_to_All || all2allv_loads == nullptr) {
        return msg_size;
    }

    auto it = all2allv_loads->find(std::make_pair(src, dst));
    if (it == all2allv_loads->end()) {
        return msg_size;
    }

    uint64_t pair_total = it->second;
    if (pair_total == 0) {
        return 1;
    }
    if (all2allv_total_size == 0 || all2allv_chunk_size == 0) {
        return pair_total;
    }

    long double scaled = static_cast<long double>(pair_total) *
                         static_cast<long double>(all2allv_chunk_size) /
                         static_cast<long double>(all2allv_total_size);
    return std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(scaled)));
}

bool AllToAllv::ready() {
    if (stream->state == StreamState::Created ||
        stream->state == StreamState::Ready) {
        stream->changeState(StreamState::Executing);
    }
    if (packets.size() == 0 || stream_count == 0 || free_packets == 0) {
        return false;
    }

    MyPacket packet = packets.front();
    uint64_t send_size = pair_msg_size(id, packet.preferred_dest);
    uint64_t recv_size = pair_msg_size(packet.preferred_src, id);

    sim_request snd_req;
    snd_req.srcRank = id;
    snd_req.dstRank = packet.preferred_dest;
    snd_req.tag = stream->stream_id;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    stream->owner->front_end_sim_send(
        0, Sys::dummy_data, send_size, UINT8, packet.preferred_dest,
        stream->stream_id, &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent, nullptr);

    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
        stream, stream->owner->id, EventType::PacketReceived,
        packet.preferred_vnet, packet.stream_id);
    stream->owner->front_end_sim_recv(
        0, Sys::dummy_data, recv_size, UINT8, packet.preferred_src,
        stream->stream_id, &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent, ehd);
    reduce();
    return true;
}
