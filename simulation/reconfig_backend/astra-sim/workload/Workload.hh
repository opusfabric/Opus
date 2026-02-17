/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __WORKLOAD_HH__
#define __WORKLOAD_HH__

#include <memory>
#include <string>
#include <unordered_map>

#include "astra-sim/workload/Scheduler.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Statistics.hh"
#include "astra-sim/workload/LocalMemUsageTracker.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"


namespace AstraSim {

class Sys;
class DataSet;
class Scheduler;

class Workload : public Callable {
  public:
    Workload(Sys* sys,
       std::string et_filename,
       std::string comm_group_filename,
       std::string provision_config,
       std::map<int,std::vector<int>>& comm_to_topo);
    ~Workload();

    // communicator groups
    // Parse the user provided 'comm_group_filename' and extract the list of
    // communicator groups. Refer to the wiki for the format.
    void initialize_comm_groups(std::string comm_group_filename);
    void issue_pytorch_pg_metadata(
        std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);

    // event-based simulation
    void issue_dep_free_nodes();
    bool issue(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_metadata(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_replay(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_remote_mem(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void issue_comp(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    bool issue_comm(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    bool issue_coll_comm(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    bool issue_send_comm(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    bool issue_recv_comm(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void skip_invalid(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void call(EventType event, CallData* data);
    void fire();

    // stats
    void report();

    Chakra::ETFeeder* et_feeder;
    std::unordered_map<int, CommunicatorGroup*> comm_groups;
    HardwareResource* hw_resource;
    Sys* sys;
    Statistics* stats;
    std::unique_ptr<LocalMemUsageTracker> local_mem_usage_tracker;
    std::unordered_map<int, uint64_t> collective_comm_node_id_map;
    std::unordered_map<int, DataSet*> collective_comm_wrapper_map;
    bool is_finished;
    Scheduler* scheduler;

    static std::map<int, int> group_vote_count;
    static std::map<int, std::map<int, int>> group_node_vote_count;
    static std::map<int, int> group_finish_count;
    static std::map<int, std::map<int, int>> group_node_finish_count;
    static std::map<int, int> group_leader_npu_id;

    static std::map<int, int> vote_rounds;
    static std::map<int, int> finish_rounds;

    static int num_bw_matrix;

    void vote(int comm_group_id);
    void finish(int comm_group_id);

    bool check_vote(int comm_group_id);
    bool check_finish(int comm_group_id);

  private:
    // From the ET node, find out the corresponding communicator group, and
    // return the pointer. If no communicator group is specified for this ET
    // node, return nullptr.
    std::vector<int> comm_group_list;
    int current_comm_group_idx;

    std::map<int,std::vector<int>>& comm_to_topo;

    int cached_reconfig_topo_id;

    CommunicatorGroup* extract_comm_group(
        std::shared_ptr<Chakra::ETFeederNode> node);
    int previous_group_id = 0;
};

}  // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
