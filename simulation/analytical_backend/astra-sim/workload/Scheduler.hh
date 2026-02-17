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
};
}

#endif // SCHEDULER_HH