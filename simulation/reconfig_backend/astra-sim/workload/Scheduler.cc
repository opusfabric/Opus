#include "astra-sim/workload/Scheduler.hh"

// #include "astra-sim/common/Logging.hh"
// #include "astra-sim/system/IntData.hh"
// #include "astra-sim/system/MemEventHandlerData.hh"
// #include "astra-sim/system/RecvPacketEventHandlerData.hh"
// #include "astra-sim/system/SendPacketEventHandlerData.hh"
// #include "astra-sim/system/WorkloadLayerHandlerData.hh"
// #include <json/json.hpp>

#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

std::vector<int> AstraSim::Scheduler::stage_dp_or_pp = std::vector<int>();
int AstraSim::Scheduler::pp_stages = 0;
int AstraSim::Scheduler::num_npu_per_pp = 0;

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

    int topo_id = 0;
    for (int i = 0; i < stage_dp_or_pp.size(); i++){
        topo_id += stage_dp_or_pp[i] << (pp_stages - 1 - i);
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
    int topo_id = 0;


    if (prev_comm_group_id != -1) {
        topo_id = 1; // PP
    } else {
        topo_id = 0;
    }

    int this_pp_group = this->sys->id / AstraSim::Scheduler::num_npu_per_pp;

    int dst_pp_group = cur_comm_group_id / AstraSim::Scheduler::num_npu_per_pp;

    // if (sys->comm_NI->get_inflight_coll() > 0) {
    //     std::cout << "Scheduler: " << this->sys->id << " cannot reconfigure now, inflight collective communication exists." << std::endl;
    //     return false;
    // }


    stage_dp_or_pp[this_pp_group] = topo_id;
    if (topo_id == 1) {
        stage_dp_or_pp[dst_pp_group] = 1;
        std::cout << "Scheduler: " << this->sys->id << " setting recv stage " << dst_pp_group << " to PP" << std::endl;
    }
    std::cout << "Scheduler: " << this->sys->id << " setting stage " << this_pp_group << " to " << (topo_id == 0 ? "DP" : "PP") << std::endl;
    std::cout << "Current stage_dp_or_pp: ";
    for (const auto& val : stage_dp_or_pp) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    

    bool can_config = this->reconfigure(cur_comm_group_id, prev_comm_group_id, skip_inflight);
    if (!can_config) return false;
        
    printf("\033[1;33mScheduler: %d Switching to comm group: %d\033[0m\n", this->sys->id, cur_comm_group_id);
    // }

    return true;
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
                std::cout << "Scheduler: " << this->sys->id << " found next target comm group id: " << target_comm_group_id << " at start idx: " << start_idx << std::endl;
                break;
            }
        }
    }

    printf("\033[1;33mScheduler: %d, current comm idx: %d, target comm group id: %d\033[0m\n", this->sys->id, this->cur_comm_idx, target_comm_group_id);
    printf("\033[1;33mScheduler: %d, current comm group id: %d\033[0m\n", this->sys->id, cur_comm_group_id);

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
        if (can_config) {
            printf("\033[1;33mScheduler: %d provision success for %d\033[0m\n", this->sys->id, target_comm_group_id);
        } else {
            printf("\033[1;33mScheduler: %d provision failed for %d\033[0m\n", this->sys->id, target_comm_group_id);
        }
    } 

    return true;
}