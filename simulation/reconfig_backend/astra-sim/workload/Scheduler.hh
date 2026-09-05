#ifndef SCHEDULER_HH
#define SCHEDULER_HH

#include <iostream>
#include <vector>
#include <unordered_map>

#include "astra-sim/system/Callable.hh"

#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/system/Sys.hh"
#include <yaml-cpp/yaml.h>
#include <fstream>


namespace AstraSim {

class Scheduler {
private:
    AstraSim::Sys* sys;
    std::unordered_map<int, std::vector<int>> comm_group_to_start_indexes;

    int cur_comm_idx = -1;
    bool reconfigure(int cur_comm_group_id, int prev_comm_group_id, int skip_inflight = 0);

public:
    Scheduler(){};
    Scheduler(AstraSim::Sys* system, std::string provision_config);
    bool pre_reconfig(int cur_comm_group_id, int prev_comm_group_id, int skip_inflight = 0);
    bool post_reconfig(int cur_comm_group_id);

    static std::vector<int> stage_dp_or_pp;
    static int pp_stages;
    static int num_npu_per_pp;
    // Figure 12/13/15 schedules use one bit per PP stage. The expert-
    // parallel Figure 14 schedules use one decimal digit per stage.
    static bool decimal_topology_ids;
    // When non-zero, the decimal digit used for EP-stage topology state.
    // The full topology ID is assembled from per-PP-stage digits.
    static int ep_topo_id;
};
}

#endif // SCHEDULER_HH