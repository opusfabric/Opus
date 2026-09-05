#include "astra-sim/workload/Scheduler.hh"

// #include "astra-sim/common/Logging.hh"
// #include "astra-sim/system/IntData.hh"
// #include "astra-sim/system/MemEventHandlerData.hh"
// #include "astra-sim/system/RecvPacketEventHandlerData.hh"
// #include "astra-sim/system/SendPacketEventHandlerData.hh"
// #include "astra-sim/system/WorkloadLayerHandlerData.hh"
// #include <json/json.hpp>

#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

std::vector<int> AstraSim::Scheduler::stage_dp_or_pp = std::vector<int>();
int AstraSim::Scheduler::pp_stages = 0;
int AstraSim::Scheduler::num_npu_per_pp = 0;
bool AstraSim::Scheduler::decimal_topology_ids = false;
int AstraSim::Scheduler::ep_topo_id = 0;

namespace {
int topo_id_from_stage_state(const std::vector<int>& stage_state) {
    int topo_id = 0;
    for (int i = 0; i < (int)stage_state.size(); i++) {
        int p = 1;
        for (int j = 0; j < (int)stage_state.size() - 1 - i; j++) p *= 10;
        topo_id += stage_state[i] * p;
    }
    return topo_id;
}

struct PendingTopoRequest {
    int stage;
    int dim;
    int dst_stage;
    unsigned long seq = 0;
};

bool same_request(const PendingTopoRequest& a, const PendingTopoRequest& b) {
    return a.stage == b.stage && a.dim == b.dim && a.dst_stage == b.dst_stage;
}

std::vector<PendingTopoRequest>& pending_topo_requests() {
    static std::vector<PendingTopoRequest> pending;
    return pending;
}

unsigned long& next_topo_request_seq() {
    static unsigned long seq = 0;
    return seq;
}

// Batching window for topology requests. Sized to cover the largest studied
// reconfiguration latency so every latency variant collects the same request
// batches and follows the same topology trajectory; runs then differ only by
// the reconfiguration cost itself, keeping latency sweeps monotonic.
constexpr Tick kTopoRequestDebounceNs = 10000000;

Tick& pending_topo_ready_time() {
    static Tick ready_time = 0;
    return ready_time;
}

void add_pending_request(const PendingTopoRequest& request) {
    auto& pending = pending_topo_requests();
    for (const auto& existing : pending) {
        if (same_request(existing, request)) return;
    }
    PendingTopoRequest stamped = request;
    stamped.seq = next_topo_request_seq()++;
    pending.push_back(stamped);
}

// Deterministic priority by request class so batch composition depends only
// on the pending set (micro-timing independent across reconfig latencies):
// EP first (a stage's continuous DP stream must not starve its EP phases),
// then PP, then DP.
int dim_priority(int dim) {
    if (dim == 3) return 0;  // EP
    if (dim == 0) return 1;  // PP
    return 2 + dim;          // DP (1), then any higher dims
}

bool request_less(const PendingTopoRequest& a, const PendingTopoRequest& b) {
    if (a.stage != b.stage) return a.stage < b.stage;
    int pa = dim_priority(a.dim);
    int pb = dim_priority(b.dim);
    if (pa != pb) return pa < pb;
    return a.dst_stage < b.dst_stage;
}

bool request_satisfied(
    const PendingTopoRequest& request,
    const std::vector<int>& stage_state) {
    if (request.stage < 0 || request.stage >= (int)stage_state.size()) return true;
    if (stage_state[request.stage] != request.dim) return false;
    if (request.dim == 0 &&
        request.dst_stage >= 0 &&
        request.dst_stage < (int)stage_state.size()) {
        return stage_state[request.dst_stage] == 0;
    }
    return true;
}

std::vector<int> apply_request(
    const std::vector<int>& stage_state,
    const PendingTopoRequest& request) {
    std::vector<int> next_stage_state = stage_state;
    next_stage_state[request.stage] = request.dim;
    if (request.dim == 0 &&
        request.dst_stage >= 0 &&
        request.dst_stage < (int)next_stage_state.size()) {
        next_stage_state[request.dst_stage] = 0;
    }
    return next_stage_state;
}

std::vector<int> request_stages(const PendingTopoRequest& request) {
    std::vector<int> stages;
    if (request.stage >= 0) {
        stages.push_back(request.stage);
    }
    if (request.dim == 0 && request.dst_stage >= 0 && request.dst_stage != request.stage) {
        stages.push_back(request.dst_stage);
    }
    return stages;
}

bool touches_marked_stage(
    const PendingTopoRequest& request,
    const std::vector<bool>& touched) {
    for (int stage : request_stages(request)) {
        if (stage >= 0 && stage < (int)touched.size() && touched[stage]) {
            return true;
        }
    }
    return false;
}

void mark_touched_stages(
    const PendingTopoRequest& request,
    std::vector<bool>& touched) {
    for (int stage : request_stages(request)) {
        if (stage >= 0 && stage < (int)touched.size()) {
            touched[stage] = true;
        }
    }
}
}

Scheduler::Scheduler(AstraSim::Sys* system, std::string provision_config) : sys(system) {
    int rank = this->sys->id;
    
    try {
        //TODO parse input from command line argument

        YAML::Node config = YAML::LoadFile(provision_config);
        if (config[std::to_string(rank)]) {
            for (const auto& group : config[std::to_string(rank)]) {
                int group_id = group.first.as<int>();
                std::vector<int> start_indexes = group.second.as<std::vector<int>>();
                comm_group_to_start_indexes[group_id] = start_indexes;
                printf("Rank: %d, Group ID: %d ", rank, group_id);
            }
        }

        // Print the comm_group_to_start_indexes for debugging
        std::cout << "Rank " << rank << " comm_group_to_start_indexes:" << std::endl;
        for (const auto& pair : comm_group_to_start_indexes) {
            std::cout << "Group ID: " << pair.first << " Start Indexes: ";
            for (const auto& index : pair.second) {
                std::cout << index << " ";
            }
            std::cout << std::endl;
        }
    } catch (const YAML::Exception& e) {
        std::cout << "Provisioning Disabled for rank " << rank << e.what() << std::endl;
    }
}

bool AstraSim::Scheduler::reconfigure(int cur_comm_group_id, int prev_comm_group_id, int skip_inflight) {
    // TODO use suitable topo_id

    // int topo_id = 0;
    // if (cur_comm_group_id != -1) {
    //     topo_id = 0; // default topology
    // } else {
    //     topo_id = 1;
    // }

    // Decimal-digit encoding: each stage gets one decimal digit.
    // digit 0 = PP (asymmetric), 1 = DP, 2 = CP, 3 = EP, ...
    // topo_id = d0 * 10^(S-1) + d1 * 10^(S-2) + ... + d_{S-1} * 10^0
    int topo_id = 0;
    if (decimal_topology_ids) {
        topo_id = topo_id_from_stage_state(stage_dp_or_pp);
    } else {
        for (int i = 0; i < (int)stage_dp_or_pp.size(); i++) {
            topo_id += stage_dp_or_pp[i] << (pp_stages - 1 - i);
        }
    }
    bool can_config = this->sys->comm_NI->sim_reconfig(topo_id, skip_inflight);
    if (!can_config) {
        //printf("\033[1;33mScheduler: %d Switching to comm group failed: %d\033[0m\n", this->sys->id, cur_comm_group_id);
        //this->sys->comm_NI->print_on_going_comms();
        return false;
    }
    return true;
}

bool AstraSim::Scheduler::pre_reconfig(int cur_comm_group_id, int prev_comm_group_id, int skip_inflight) {
                                            // Used for dst id in PP

    // Legacy schedules encode each pipeline stage as one bit: 0 selects DP
    // and 1 selects PP. Figure 14 uses decimal digits to also represent EP.
    if (!decimal_topology_ids) {
        int stage_mode = (prev_comm_group_id == -1) ? 0 : 1;
        int this_pp_group = this->sys->id / num_npu_per_pp;
        stage_dp_or_pp[this_pp_group] = stage_mode;
        if (stage_mode == 1) {
            int dst_pp_group = cur_comm_group_id / num_npu_per_pp;
            if (dst_pp_group >= 0 && dst_pp_group < pp_stages) {
                stage_dp_or_pp[dst_pp_group] = 1;
            }
        }
        return this->reconfigure(cur_comm_group_id, prev_comm_group_id,
                                 skip_inflight);
    }

    // Dimension-digit mapping (prev_comm_group_id carries the digit value):
    //   -1  -> 1 (DP, default symmetric, backward-compatible)
    //    0  -> 0 (PP, asymmetric pipeline send/recv)
    //   >0  -> used as-is (1=DP, 2=CP, 3=EP, ...)
    int dim_digit = (prev_comm_group_id < 0) ? 1 : prev_comm_group_id;

    int this_pp_group = this->sys->id / AstraSim::Scheduler::num_npu_per_pp;
    int dst_pp_group  = cur_comm_group_id / AstraSim::Scheduler::num_npu_per_pp;
    PendingTopoRequest current_request{
        this_pp_group,
        dim_digit,
        dim_digit == 0 ? dst_pp_group : -1,
    };
    add_pending_request(current_request);

    auto& pending = pending_topo_requests();
    std::sort(pending.begin(), pending.end(), request_less);

    bool current_satisfied = request_satisfied(current_request, stage_dp_or_pp);
    for (auto it = pending.begin(); it != pending.end();) {
        if (request_satisfied(*it, stage_dp_or_pp)) {
            bool was_current = same_request(*it, current_request);
            it = pending.erase(it);
            if (was_current) current_satisfied = true;
            continue;
        }
        ++it;
    }
    if (pending.empty()) {
        pending_topo_ready_time() = 0;
        return current_satisfied;
    }
    if (current_satisfied) {
        // The current topology already serves this request: let the work
        // proceed (it will hold the inflight gate while it runs). Pending
        // flips are triggered by retries of the still-unsatisfied ranks.
        for (auto it = pending.begin(); it != pending.end(); ++it) {
            if (same_request(*it, current_request)) {
                pending.erase(it);
                break;
            }
        }
        return true;
    }

    Tick now = AstraSim::Sys::boostedTick();
    Tick& ready_time = pending_topo_ready_time();
    if (ready_time == 0 || now > ready_time + kTopoRequestDebounceNs) {
        ready_time = now + kTopoRequestDebounceNs;
    }
    if (now < ready_time) {
        return false;
    }

    std::vector<int> next_stage_state = stage_dp_or_pp;
    std::vector<bool> touched(next_stage_state.size(), false);
    std::vector<PendingTopoRequest> selected_requests;
    for (const auto& request : pending) {
        if (request_satisfied(request, next_stage_state)) {
            continue;
        }
        if (touches_marked_stage(request, touched)) {
            continue;
        }
        next_stage_state = apply_request(next_stage_state, request);
        mark_touched_stages(request, touched);
        selected_requests.push_back(request);
    }
    if (selected_requests.empty()) return current_satisfied;

    if (this->sys->trace_enabled) {
        std::cout << "Scheduler: " << this->sys->id << " selected";
        for (const auto& request : selected_requests) {
            std::cout << " [stage " << request.stage << " -> dim " << request.dim << "]";
        }
        std::cout << std::endl;
        std::cout << "Current stage_dp_or_pp: ";
        for (const auto& val : next_stage_state) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    int topo_id = topo_id_from_stage_state(next_stage_state);
    bool can_config = this->sys->comm_NI->sim_reconfig(topo_id, skip_inflight);
    if (!can_config) return false;

    stage_dp_or_pp = next_stage_state;
    for (auto it = pending.begin(); it != pending.end();) {
        if (request_satisfied(*it, stage_dp_or_pp)) {
            it = pending.erase(it);
        } else {
            ++it;
        }
    }
    pending_topo_ready_time() =
        pending.empty() ? 0 : AstraSim::Sys::boostedTick() + kTopoRequestDebounceNs;

    if (this->sys->trace_enabled) {
        printf("Scheduler: %d Switching to comm group: %d\n", this->sys->id, cur_comm_group_id);
    }
    // Success means the new topology serves this request — whether it was
    // applied just now or was already compatible (and therefore skipped
    // during selection).
    return request_satisfied(current_request, stage_dp_or_pp);
}

bool AstraSim::Scheduler::post_reconfig(int cur_comm_group_id) {
                                  //  -1: Finished PP, other: Finished DP
    // find the next start idx closest to current cur_comm_idx
    int closest_start_idx = std::numeric_limits<int>::max();
    int target_comm_group_id = cur_comm_group_id;


    for (const auto& pair : comm_group_to_start_indexes) {
        int group_id = pair.first;
        const std::vector<int>& start_indexes = pair.second;

        for (int start_idx : start_indexes) {
            if (start_idx == this->cur_comm_idx) {
                closest_start_idx = start_idx;
                target_comm_group_id = group_id;
                if (this->sys->trace_enabled) {
                    std::cout << "Scheduler: " << this->sys->id << " found next target comm group id: " << target_comm_group_id << " at start idx: " << start_idx << std::endl;
                }
                break;
            }
        }
    }

    if (this->sys->trace_enabled) {
        printf("\033[1;33mScheduler: %d, current comm idx: %d, target comm group id: %d\033[0m\n", this->sys->id, this->cur_comm_idx, target_comm_group_id);
        printf("\033[1;33mScheduler: %d, current comm group id: %d\033[0m\n", this->sys->id, cur_comm_group_id);
    }

    // TODO increment for PP as well
    this->cur_comm_idx++;
    
    // reconfigure
    if (target_comm_group_id != cur_comm_group_id) {

        int type = -1;
        if (target_comm_group_id != -1) {
            type = 1; // Finished PP
        } 

        bool can_config = this->pre_reconfig(target_comm_group_id, type);

        // bool can_config = this->reconfigure(target_comm_group_id, cur_comm_group_id);
        if (this->sys->trace_enabled) {
            if (can_config) {
                printf("\033[1;33mScheduler: %d provision success for %d\033[0m\n", this->sys->id, target_comm_group_id);
            } else {
                printf("\033[1;33mScheduler: %d provision failed for %d\033[0m\n", this->sys->id, target_comm_group_id);
            }
        }
    } 

    return true;
}