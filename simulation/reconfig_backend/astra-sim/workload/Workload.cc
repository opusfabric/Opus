/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/Workload.hh"

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/MemEventHandlerData.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/Scheduler.hh"
#include <json/json.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;
using namespace AstraSim;
using namespace Chakra::FeederV3;
using json = nlohmann::json;

typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

namespace {
bool is_contiguous_comm_group(const CommunicatorGroup* comm_group) {
    if (comm_group == nullptr || comm_group->involved_NPUs.empty()) {
        return false;
    }

    std::vector<int> ranks = comm_group->involved_NPUs;
    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
    if (ranks.size() != comm_group->involved_NPUs.size()) {
        return false;
    }

    return ranks.back() - ranks.front() + 1 == static_cast<int>(ranks.size());
}

bool is_topology_agnostic_tp_group(
    const CommunicatorGroup* comm_group,
    const std::vector<int>& compatible_topos,
    int num_topologies,
    bool explicit_reconfig) {
    if (explicit_reconfig ||
        compatible_topos.size() != static_cast<size_t>(num_topologies)) {
        return false;
    }
    return !AstraSim::Scheduler::decimal_topology_ids ||
           is_contiguous_comm_group(comm_group);
}
}  // namespace

std::map<int, int> Workload::group_vote_count = std::map<int, int>();
std::map<int, int> Workload::group_finish_count = std::map<int, int>();
std::map<int, int> Workload::group_leader_npu_id = std::map<int, int>();

std::map<int, int> Workload::vote_rounds = std::map<int, int>();
std::map<int, int> Workload::finish_rounds = std::map<int, int>();

std::map<int, std::map<int, int>> Workload::group_node_vote_count = std::map<int, std::map<int, int>>();
std::map<int, std::map<int, int>> Workload::group_node_finish_count = std::map<int, std::map<int, int>>();

int Workload::num_bw_matrix = 0;

void Workload::vote(int comm_group_id) {
    if (group_node_vote_count.find(comm_group_id) ==
        group_node_vote_count.end()) {
        group_node_vote_count[comm_group_id] = std::map<int, int>();
    }

    if (group_node_finish_count.find(comm_group_id) ==
        group_node_finish_count.end()) {
        group_node_finish_count[comm_group_id] = std::map<int, int>();
    }

    if (group_node_vote_count[comm_group_id].find(this->sys->id) ==
        group_node_vote_count[comm_group_id].end()) {
        group_node_vote_count[comm_group_id][this->sys->id] = 0;
    }

    if (group_node_finish_count[comm_group_id].find(this->sys->id) ==
        group_node_finish_count[comm_group_id].end()) {
        group_node_finish_count[comm_group_id][this->sys->id] = 0;
    }

    if (vote_rounds.find(comm_group_id) == vote_rounds.end()) {
        vote_rounds[comm_group_id] = 0;
    }

    if (finish_rounds.find(comm_group_id) == finish_rounds.end()) {
        finish_rounds[comm_group_id] = 0;
    }

    if (group_node_vote_count[comm_group_id][this->sys->id] >
        group_node_finish_count[comm_group_id][this->sys->id]) {
        if (sys->trace_enabled) {
            std::cout << "RANK: " << this->sys->id << " has already voted for comm group "
                     << comm_group_id << " in this round." << std::endl;
        }
    }
    else{
        group_node_vote_count[comm_group_id][this->sys->id] += 1;
    }
    if (sys->trace_enabled) {
        std::cout << "RANK: " << this->sys->id << " Vote for comm group "
                  << comm_group_id << ", current node vote count: "
                  << group_node_vote_count[comm_group_id][this->sys->id]
                  << std::endl;
    }
}

void Workload::finish(int comm_group_id) {
    if (group_node_finish_count.find(comm_group_id) ==
        group_node_finish_count.end()) {
        group_node_finish_count[comm_group_id] = std::map<int, int>();
    }

    if (finish_rounds.find(comm_group_id) == finish_rounds.end()) {
        finish_rounds[comm_group_id] = 0;
    }

    if (group_node_finish_count[comm_group_id].find(this->sys->id) ==
        group_node_finish_count[comm_group_id].end()) {
        group_node_finish_count[comm_group_id][this->sys->id] = 0;
    }

    group_node_finish_count[comm_group_id][this->sys->id] += 1;
    if (sys->trace_enabled) {
        std::cout << "RANK: " << this->sys->id << " Finish for comm group "
                  << comm_group_id << ", current node finish count: "
                  << group_node_finish_count[comm_group_id][this->sys->id]
                  << std::endl;
    }
}

bool Workload::check_vote(int comm_group_id) {
    int required_votes = comm_groups[comm_group_id]->involved_NPUs.size();
    int current_votes = 0;
    if (group_node_vote_count[comm_group_id][this->sys->id] - finish_rounds[comm_group_id] > 1) {
        if (sys->trace_enabled) {
            std::cout << "RANK: " << this->sys->id << " is ahead, previous comm has not finished yet." << std::endl;
        }
        return false;
    }


    if (sys->trace_enabled) {
        std::cout << "RANK: " << this->sys->id << " checking votes for comm group " << comm_group_id << ": ";
    }
    if (group_node_vote_count.find(comm_group_id) !=
        group_node_vote_count.end()) {
        for (const auto& pair : group_node_vote_count[comm_group_id]) {
            if (pair.second > finish_rounds[comm_group_id]) {
                current_votes += 1;
                if (sys->trace_enabled) {
                    std::cout << pair.first << " ";
                }
            }
        }
    }

    if (sys->trace_enabled) {
        std::cout << " -- (" << current_votes << "/" << required_votes << ") [" << finish_rounds[comm_group_id] << "]\n";
    }
    return current_votes >= required_votes;
}

bool Workload::check_finish(int comm_group_id) {
    int required_finishes = comm_groups[comm_group_id]->involved_NPUs.size();
    int current_finishes = 0;
    if (sys->trace_enabled) {
        std::cout << "RANK: " << this->sys->id << " checking finishes for comm group " << comm_group_id << ": ";
    }
    if (group_node_finish_count.find(comm_group_id) !=
        group_node_finish_count.end()) {
        for (const auto& pair : group_node_finish_count[comm_group_id]) {
            if (pair.second > finish_rounds[comm_group_id]) {
                current_finishes += 1;
                if (sys->trace_enabled) {
                    std::cout << pair.first << " ";
                }
            }
        }
    }

    if (sys->trace_enabled) {
        std::cout << " -- (" << current_finishes << "/" << required_finishes << ") [" << finish_rounds[comm_group_id] << "]\n";
    }
    return current_finishes >= required_finishes;
}


Workload::Workload(Sys* sys, string et_filename, string comm_group_filename, string provision_config, std::map<int,std::vector<int>>& comm_to_topo) :
    comm_to_topo(comm_to_topo) {
    string workload_filename = et_filename + "." + to_string(sys->id) + ".et";
    // Check if workload filename exists
    if (access(workload_filename.c_str(), R_OK) < 0) {
        string error_msg;
        if (errno == ENOENT) {
            error_msg =
                "workload file: " + workload_filename + " does not exist";
        } else if (errno == EACCES) {
            error_msg = "workload file: " + workload_filename +
                        " exists but is not readable";
        } else {
            error_msg =
                "Unknown workload file: " + workload_filename + " access error";
        }
        LoggerFactory::get_logger("workload")->critical(error_msg);
        exit(EXIT_FAILURE);
    }
    auto temp_et_feeder = new ETFeeder(workload_filename);
    this->comm_group_list = temp_et_feeder->traverse_comm_group();
    this->current_comm_group_idx = 0;
    delete temp_et_feeder;
    this->et_feeder = new ETFeeder(workload_filename);
    this->comm_groups.clear();
    // TODO: parametrize the number of available hardware resources
    this->hw_resource = new HardwareResource(1, sys->id);
    this->local_mem_usage_tracker =
        std::make_unique<LocalMemUsageTracker>(sys->id);
    this->sys = sys;
    size_t group_dir_end = comm_group_filename.find_last_of("/\\");
    if (group_dir_end == std::string::npos) {
        this->all2allv_load_filename = "all2allv_loads.json";
    } else {
        this->all2allv_load_filename =
            comm_group_filename.substr(0, group_dir_end + 1) +
            "all2allv_loads.json";
    }
    initialize_comm_groups(comm_group_filename);
    this->stats = new Statistics(this);
    this->is_finished = false;

    this->scheduler = new Scheduler(sys, provision_config);
    // choose a non-relavant comm group to trigger initial reconfiguration
    this->scheduler->post_reconfig(-10);

    std::cout << "Workload::Workload, sys->id=" << sys->id << " Comm groups: ";
    for(int comm_group_id : this->comm_group_list) {
        std::cout << comm_group_id << " ";
    }
    std::cout << std::endl;
}

Workload::~Workload() {
    for (auto comm_group : comm_groups) {
        delete comm_group.second;
    }
    comm_groups.clear();

    if (this->et_feeder != nullptr) {
        delete this->et_feeder;
    }
    if (this->hw_resource != nullptr) {
        delete this->hw_resource;
    }
    if (this->stats != nullptr) {
        delete this->stats;
    }
    if (this->scheduler != nullptr) {
        delete this->scheduler;
    }
}

void Workload::initialize_comm_groups(string comm_group_filename) {
    // communicator group input file is not given
    if (comm_group_filename.find("empty") != std::string::npos) {
        comm_groups.clear();
        return;
    }

    ifstream inFile;
    json j;
    inFile.open(comm_group_filename);
    inFile >> j;

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        std::string comm_group_name = it.key();
        int comm_group_id = std::stoi(comm_group_name);

        std::vector<int> involved_NPUs;
        for (auto id : it.value()) {
            involved_NPUs.push_back(id);
        }

        comm_groups[comm_group_id] =
            new CommunicatorGroup(comm_group_id, involved_NPUs, sys);
    }
}

void Workload::issue_pytorch_pg_metadata(
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    // For read comm groups from torch, might overwrite previous.
    std::string pg_info = node->get_inputs_values();
    if (pg_info.empty()) {
        return;
    }
    pg_info = pg_info.substr(2, pg_info.size() - 4);

    try {
        json valuesRoot = json::parse(pg_info);

        for (const auto& item : valuesRoot) {
            std::string pgName = item.at("pg_name").get<std::string>();
            std::vector<int> involved_NPUs =
                item.at("ranks").get<std::vector<int>>();

            if (involved_NPUs.empty()) {
                for (int i = 0; i < sys->total_nodes; i++) {
                    involved_NPUs.push_back(i);
                }
            }

            int32_t pgNameInt = std::stoi(pgName);
            // To ensure pgName > 0
            CommunicatorGroup* cg =
                new CommunicatorGroup(pgNameInt + 1, involved_NPUs, sys);
            this->comm_groups[pgNameInt] = cg;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing or processing JSON: " << e.what()
                  << std::endl;
    }
}

void Workload::issue_dep_free_nodes() {
    auto& dependancy_resolver = this->et_feeder->getDependancyResolver();
    auto dependancy_free_nodes =
        dependancy_resolver.get_dependancy_free_nodes();
    std::set<uint64_t> dependancy_free_nodes_set;
    for (const auto node_id : dependancy_free_nodes) {
        dependancy_free_nodes_set.insert(node_id);
    }

    // std::cout << "Workload::issue_dep_free_nodes, sys->id=" << sys->id
    //           << ", tick=" << Sys::boostedTick()
    //           << ", dependancy_free_nodes_set.size="
    //           << dependancy_free_nodes_set.size() << std::endl;

    bool success = true;
    for (const auto node_id : dependancy_free_nodes_set) {
        std::shared_ptr<ETFeederNode> node = et_feeder->lookupNode(node_id);
        if (hw_resource->is_available(node)) {
            success = issue(node);
            // if (!success) {
            //     std::cout << "Workload::issue failed, sys->id=" << sys->id 
            //               << ", node->id=" << node->id()
            //               << ", node->name=" << node->name()
            //               << ", node->type=" << static_cast<uint64_t>(node->type())
            //               << std::endl;
            // }
        }
    }
}

bool Workload::issue_reconfig(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->type() == ChakraNodeType::COMP_NODE && node->has_attr("reconfig_dim")) {
        int dim = node->get_attr<int>("reconfig_dim");
        int skip_inflight = node->get_attr<int>("reconfig_skip_inflight", 0);
        bool can_config = this->scheduler->pre_reconfig(-1, dim, skip_inflight);
        if (!can_config) {
            if (this->sys->trace_enabled) {
                std::cout << "Workload: " << this->sys->id
                          << " stage-local reconfig to dim " << dim
                          << " blocked before node " << node->id()
                          << " (" << node->name() << ")" << std::endl;
            }
            timespec_t delta = {NS, 100000.0};
            this->sys->comm_NI->sim_schedule(delta, &Sys::handleEvent,
                new BasicEventHandlerData(this->sys->id, EventType::CallEvents));
            return false;
        }

        if (this->sys->trace_enabled) {
            std::cout << "Workload: " << this->sys->id
                      << " stage-local reconfig to dim " << dim
                      << " before node " << node->id()
                      << " (" << node->name() << ")" << std::endl;
        }
        return true;
    }

    if (!node->has_attr("reconfig_topo_id")) {
        return true;
    }

    int topo_id = node->get_attr<int>("reconfig_topo_id");
    int skip_inflight = node->get_attr<int>("reconfig_skip_inflight", 0);
    bool can_config = this->sys->comm_NI->sim_reconfig(topo_id, skip_inflight);
    if (!can_config) {
        if (this->sys->trace_enabled) {
            std::cout << "Workload: " << this->sys->id
                      << " explicit reconfig to topo_id " << topo_id
                      << " blocked before node " << node->id()
                      << " (" << node->name() << ")" << std::endl;
        }
        timespec_t delta = {NS, 100000.0};
        this->sys->comm_NI->sim_schedule(delta, &Sys::handleEvent,
            new BasicEventHandlerData(this->sys->id, EventType::CallEvents));
        return false;
    }

    // Keep the scheduler's per-stage digit state coherent with the explicit
    // jump, or later dim-based requests would evaluate against stale digits.
    {
        auto& stage_state = AstraSim::Scheduler::stage_dp_or_pp;
        int value = topo_id;
        for (int i = (int)stage_state.size() - 1; i >= 0; i--) {
            stage_state[i] = value % 10;
            value /= 10;
        }
    }

    if (this->sys->trace_enabled) {
        std::cout << "Workload: " << this->sys->id
                  << " explicit reconfig to topo_id " << topo_id
                  << " before node " << node->id()
                  << " (" << node->name() << ")" << std::endl;
    }
    return true;
}

bool Workload::issue(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    bool success = true;
    auto logger = LoggerFactory::get_logger("workload");
    // std::cout << "Workload::issue, sys->id=" << sys->id
    //           << ", tick=" << Sys::boostedTick()
    //           << ", node->id=" << node->id()
    //           << ", node->name=" << node->name()
    //           << ", node->type=" << static_cast<uint64_t>(node->type())
    //           << std::endl;
    
    if (sys->trace_enabled) {
        logger->debug("issue,sys->id={}, tick={}, node->id={}, "
                      "node->name={}, node->type={}",
                      sys->id, Sys::boostedTick(), node->id(), node->name(),
                      static_cast<uint64_t>(node->type()));
    }

    if (!issue_reconfig(node)) {
        return false;
    }

    this->et_feeder->getDependancyResolver().take_node(node->id());
    this->hw_resource->occupy(node);
    // stats->record_end will be called in Workload::call
    stats->record_start(node, Sys::boostedTick());
    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->recordStart(node, Sys::boostedTick());
    }
    if (sys->replay_only) {
        issue_replay(node);
    } else {
        if ((node->type() == ChakraNodeType::MEM_LOAD_NODE) ||
            (node->type() == ChakraNodeType::MEM_STORE_NODE)) {
            issue_remote_mem(node);
        } else if (node->type() == ChakraNodeType::COMP_NODE) {
            if (!this->sys->roofline_enabled) {
                issue_replay(node);

            } else {
                if (node->is_cpu_op<bool>(false)) {
                    // comp node on cpu
                    // should only appears in real system trace and should run
                    // with replay.
                    issue_replay(node);
                } else {
                    // comp node on gpu
                    issue_comp(node);
                }
            }
        } else if (node->type() == ChakraNodeType::COMM_COLL_NODE ||
                   node->type() == ChakraNodeType::COMM_SEND_NODE ||
                   node->type() == ChakraNodeType::COMM_RECV_NODE) {
            success = issue_comm(node);
            if (!success) {
                hw_resource->release(node);
                this->et_feeder->getDependancyResolver().push_back_node(node->id());
                stats->record_end(node, Sys::boostedTick());
                if (this->sys->track_local_mem) {
                    this->local_mem_usage_tracker->recordEnd(
                        node, Sys::boostedTick());
                }
                return success;
            }
        } else if (node->type() == ChakraNodeType::INVALID_NODE) {
            skip_invalid(node);
        } else if (node->type() == ChakraNodeType::METADATA_NODE) {
            issue_metadata(node);
        } else {
            logger->critical("Unknown node type");
            exit(EXIT_FAILURE);
        }
    }
    return success;
}

void Workload::issue_metadata(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    // TODO: someway to identify this metadata node is a pytorch pg node
    if (true) {
        issue_pytorch_pg_metadata(node);
    } else {
        throw std::runtime_error("Unknown metadata node type");
    }
    this->skip_invalid(node);  // for proper dependancy resolving
}

void Workload::issue_replay(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node->id();
    uint64_t runtime = 1ul;
    if (node->runtime() != 0ul) {
        // chakra runtimes are in microseconds and we should convert it into
        // nanoseconds
        runtime = node->runtime() * 1000;
    }
    if (node->is_cpu_op()) {
        hw_resource->tics_cpu_ops += runtime;
    } else {
        hw_resource->tics_gpu_ops += runtime;
    }
    // printf("Workload::issue_replay, sys->id=%d, node->id=%lu, runtime=%lu ns\n",
    //        sys->id, node->id(), runtime);
    sys->register_event(this, EventType::General, wlhd, runtime);
}

void Workload::issue_remote_mem(
    shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node->id();
    sys->remote_mem->issue(node->tensor_size(), wlhd);
}

void Workload::issue_comp(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (!this->sys->roofline_enabled) {
        throw std::runtime_error(
            "Roofline model is not enabled for non-replay comp");
    }

    if (node->is_cpu_op()) {
        throw std::runtime_error("Roofline is only available for GPU nodes");
        return;
    }

    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node->id();

    double num_ops = static_cast<double>(node->num_ops<uint64_t>());
    double tensor_size = static_cast<double>(node->tensor_size<uint64_t>());


    // if tensor_size is 0 during roofline mode, this is an invalid node
    if (tensor_size == 0) {
        skip_invalid(node);
        return;
    }

    double operational_intensity = num_ops / tensor_size;
    double perf = sys->roofline->get_perf(operational_intensity);
    double elapsed_time = static_cast<double>(node->num_ops()) / perf;  // sec
    uint64_t runtime = static_cast<uint64_t>(elapsed_time * 1e9);  // sec -> ns
    // std::cout << "Workload::issue_comp, sys->id=" << sys->id
    //           << ", node->id=" << node->id() << ", start_time=" << Sys::boostedTick()
    //           << ", runtime=" << runtime << " ns"
    //           << std::endl;

    // printf(
    //     "Workload::issue_comp, sys->id=%d, node->id=%lu, num_ops=%e, "
    //     "tensor_size=%e, operational_intensity=%f, perf=%f, elapsed_time=%f "
    //     "sec, runtime=%lu ns\n",
    //     sys->id, node->id(), num_ops, tensor_size, operational_intensity,
    //     perf, elapsed_time, runtime);
    
    if (node->is_cpu_op()) {
        hw_resource->tics_cpu_ops += runtime;
    } else {
        hw_resource->tics_gpu_ops += runtime;
    }
    sys->register_event(this, EventType::General, wlhd, runtime);

    auto& op_stat = this->stats->get_operator_statistics(node->id());
    op_stat.operation_intensity = operational_intensity;
    op_stat.compute_utilization = perf / sys->peak_perf;
    op_stat.memory_utilization =
        (perf / operational_intensity) / sys->local_mem_bw;
    op_stat.is_memory_bound = perf < sys->peak_perf;
    LoggerFactory::get_logger("workload")
        ->debug("operation_intensity={}, perf={}, elapsed_time={} "
                "compute_utilization={} memory_utilization={} tensor_size={} "
                "num_ops={}",
                operational_intensity, perf, elapsed_time,
                op_stat.compute_utilization.value(),
                op_stat.memory_utilization.value(), tensor_size, num_ops);
}

bool Workload::issue_comm(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->is_cpu_op<bool>(false)) {
        throw std::runtime_error("Comm node should not be on CPU");
    }
    bool success = true;
    const auto node_type = node->type();
    if (node_type == ChakraNodeType::COMM_COLL_NODE) {
        success = this->issue_coll_comm(node);
    } else if (node_type == ChakraNodeType::COMM_SEND_NODE) {
        success = this->issue_send_comm(node);
    } else if (node_type == ChakraNodeType::COMM_RECV_NODE) {
        success = this->issue_recv_comm(node);
    } else {
        throw std::runtime_error("Unknown comm node type");
    }
    return success;
}

bool Workload::issue_coll_comm(
    shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const bool has_involve_dims = node->has_attr("involve_dims");
    std::vector<bool> involved_dims;
    if (node->has_attr("involved_dim")) {
        const ChakraProtoMsg::AttributeProto& attr =
            node->get_attr_msg("involved_dim");

        // Ensure the attribute is of type bool_list before accessing
        if (attr.has_bool_list()) {
            const ChakraProtoMsg::BoolList& bool_list = attr.bool_list();

            // Traverse bool_list and add values to involved_dim
            for (int i = 0; i < bool_list.values_size(); ++i) {
                involved_dims.push_back(bool_list.values(i));
            }
        } else {
            cerr << "Expected bool_list in involved_dim but found another type."
                 << endl;
            exit(EXIT_FAILURE);
        }
    } else {
        // involved_dims does not exist in ETFeeder.
        // Assume involved_dims = [1,1,1,1,1] which we could simulate
        // 5-Dimension. Could use Process Group to build involved_dims later.
        // Once process group is implemented, you should get
        // that with node->pg_name()

        for (int i = 0; i < 4; i++) {
            involved_dims.push_back(true);
        }
    }

    CommunicatorGroup* comm_group = extract_comm_group(node);

    int group_size = comm_group->involved_NPUs.size();
    
    if (sys->trace_enabled) {
        std::cout << "Involved npus: ";
        for (auto d : comm_group->involved_NPUs) {
            std::cout << d << " ";
        }
        std::cout << std::endl;
    }

    // std::cout << std::endl;

    // TODO in addition to comm group detection, also check topo id change
    // Lazy reconfiguration

    int comm_id = comm_group->get_id();
    bool explicit_reconfig = node->has_attr("reconfig_topo_id");
    bool explicit_dim_reconfig = node->has_attr("reconfig_dim");
    bool skip_scheduler_reconfig =
        node->get_attr<int>("skip_scheduler_reconfig", 0) != 0;
    const auto comm_type =
        static_cast<ChakraCollectiveCommType>(node->comm_type<uint64_t>());
    bool topology_agnostic_tp =
        is_topology_agnostic_tp_group(
            comm_group, comm_to_topo[comm_id], num_bw_matrix,
            explicit_reconfig || explicit_dim_reconfig);
    // BEGIN RECONFIGURATION LOGIC
    #ifndef CONGESTION_AWARE

    if (topology_agnostic_tp || skip_scheduler_reconfig) {
        if (sys->trace_enabled && topology_agnostic_tp) {
            std::cout << "TP COMM pattern detected!" << std::endl;
        }
    } else {
        vote(comm_id);
        // Decimal EP scheduling needs explicit wakeups for ranks whose
        // dependency queues have no other pending event.
        if (AstraSim::Scheduler::decimal_topology_ids) {
            for (int npu_id : comm_group->involved_NPUs) {
                if (npu_id == this->sys->id) continue;
                timespec_t delta = {NS, 0.0};
                Sys::all_sys[npu_id]->comm_NI->sim_schedule(
                    delta, &Sys::handleEvent,
                    new BasicEventHandlerData(npu_id, EventType::CallEvents));
            }
        }
        int passed_vote = check_vote(comm_id);
        if (passed_vote) {
            if (vote_rounds[comm_id] == finish_rounds[comm_id]){
                if(topology_agnostic_tp) {
                    if (sys->trace_enabled) {
                        std::cout << "TP COMM pattern detected!" << std::endl;
                    }
                } else {
                    // Decimal-digit topo_id scheme:
                    //   dim = EP digit when this PP stage is compatible only with
                    //   EP topologies, not DP topologies. Topology IDs are decimal
                    //   states, so inspect the digit for this rank's stage instead
                    //   of treating digit 3 as a full topology ID.
                    //   dim = -1 -> 1 (DP) for all other collectives.
                    int dim = -1;  // default: DP (digit 1)
                    if (node->has_attr("reconfig_dim")) {
                        dim = node->get_attr<int>("reconfig_dim");
                    } else if (AstraSim::Scheduler::ep_topo_id != 0) {
                        const auto& topos = comm_to_topo[comm_id];
                        int stage = this->sys->id / AstraSim::Scheduler::num_npu_per_pp;
                        int pp_stages = AstraSim::Scheduler::pp_stages;
                        auto has_stage_digit = [stage, pp_stages, &topos](int digit) {
                            for (int topo_id : topos) {
                                int value = topo_id;
                                for (int i = 0; i < pp_stages - 1 - stage; i++) {
                                    value /= 10;
                                }
                                if (value % 10 == digit) {
                                    return true;
                                }
                            }
                            return false;
                        };

                        bool has_ep = has_stage_digit(AstraSim::Scheduler::ep_topo_id);
                        bool has_dp = has_stage_digit(1);
                        if (has_ep && !has_dp) {
                            dim = AstraSim::Scheduler::ep_topo_id;
                        }
                    }
                    if (!explicit_reconfig) {
                        bool reconfig_success = this->scheduler->pre_reconfig(comm_group->get_id(), dim, 0);
                        if (!reconfig_success) {
                            if (AstraSim::Scheduler::decimal_topology_ids) {
                                timespec_t delta = {NS, 100000.0};
                                this->sys->comm_NI->sim_schedule(
                                    delta, &Sys::handleEvent,
                                    new BasicEventHandlerData(
                                        this->sys->id, EventType::CallEvents));
                            }
                            return false;
                        }
                    }
                }

                group_leader_npu_id[comm_group->get_id()] = this->sys->id;
                std::cout << "RANK: " << this->sys->id << " is the leader for comm group " << comm_group->get_id() << " in round " << vote_rounds[comm_id] << std::endl;
                sys->increment_inflight_coll(this->sys->id, "COLL COMM " + std::to_string(node->id()) + " COMM GROUP " + std::to_string(comm_group->get_id()));
                vote_rounds[comm_group->get_id()] += 1;

                if (AstraSim::Scheduler::decimal_topology_ids) {
                    for (int npu_id : comm_group->involved_NPUs) {
                        if (npu_id == this->sys->id) continue;
                        Sys::all_sys[npu_id]->workload->issue_dep_free_nodes();
                        timespec_t delta = {NS, 0.0};
                        auto* ehd = new BasicEventHandlerData(
                            npu_id, EventType::CallEvents);
                        Sys::all_sys[npu_id]->comm_NI->sim_schedule(
                            delta, &Sys::handleEvent, ehd);
                    }
                }
            }
        } else {
            if (sys->trace_enabled) {
                std::cout << "RANK: " << this->sys->id << " waiting for other nodes to vote for comm group " << comm_group->get_id() << std::endl;
            }
            return false;
        }

    }

    #endif

    if (sys->trace_enabled) {
        std::cout << "\033[0;32mRANK: " << this->sys->id << " at time " << Sys::boostedTick() << " Issuing collective " << comm_group->to_string() << "\033[0m" << std::endl;
        std::cout << " COLL COMM NODE: " << node->id() << " of comm group " << comm_group->get_id() << std::endl;
        std::cout << "\033[0;32mRANK: " << this->sys->id << " inflight collective count: " << sys->get_inflight_coll() << "\033[0m" << std::endl;
    }
    
    previous_group_id = comm_id;

    const auto comm_size = node->comm_size<uint64_t>();
    // Record communication size for bandwidth calculation
    stats->get_operator_statistics(node->id()).comm_size = comm_size;
    // TODO: comm_tag? which is used to distinguish two different collective in
    // same pg
    const auto comm_priority = node->comm_priority<uint32_t>();  // default 0u


    if (comm_type == ChakraCollectiveCommType::ALL_REDUCE) {
        DataSet* fp = sys->generate_all_reduce(comm_size, involved_dims,
                                               comm_group, comm_priority);
        collective_comm_node_id_map[fp->my_id] = node->id();
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->comm_group_id = comm_group->get_id();
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
        if (sys->trace_enabled) {
            std::cout << "Generated all-reduce DataSet, fp->my_id=" << fp->my_id << std::endl;
        }
    } else if (comm_type == ChakraCollectiveCommType::ALL_TO_ALL) {
        AllToAllVLoadMapPtr all2allv_loads = nullptr;
        if (node->has_attr("all2allv_load_key")) {
            all2allv_loads = load_all2allv_loads(
                node->get_attr<std::string>("all2allv_load_key"));
        }
        DataSet* fp = sys->generate_all_to_all(comm_size, involved_dims,
                                               comm_group, comm_priority,
                                               all2allv_loads);
        fp->comm_group_id = comm_group->get_id();
        collective_comm_node_id_map[fp->my_id] = node->id();
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
        std::cout << "Generated all-to-all DataSet, fp->my_id=" << fp->my_id << std::endl;
    } else if (comm_type == ChakraCollectiveCommType::ALL_GATHER) {
        DataSet* fp = sys->generate_all_gather(comm_size, involved_dims,
                                               comm_group, comm_priority);
        fp->comm_group_id = comm_group->get_id();    
        collective_comm_node_id_map[fp->my_id] = node->id();
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
        std::cout << "Generated all-gather DataSet, fp->my_id=" << fp->my_id << std::endl;
    } else if (comm_type == ChakraCollectiveCommType::REDUCE_SCATTER) {
        DataSet* fp = sys->generate_reduce_scatter(comm_size, involved_dims,
                                                   comm_group, comm_priority);
        fp->comm_group_id = comm_group->get_id();
        collective_comm_node_id_map[fp->my_id] = node->id();
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
        std::cout << "Generated reduce-scatter DataSet, fp->my_id=" << fp->my_id << std::endl;
    } else if (comm_type == ChakraCollectiveCommType::BROADCAST) {
        // TODO: implement broadcast, for now just replay
        uint64_t runtime = 1ul;
        if (node->runtime() != 0ul) {
            // chakra runtimes are in microseconds and we should convert it into
            // nanoseconds
            runtime = node->runtime() * 1000;
        }
        DataSet* fp = new DataSet(1);
        fp->comm_group_id = comm_group->get_id();
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
        collective_comm_node_id_map[fp->my_id] = node->id();
        collective_comm_wrapper_map[fp->my_id] = fp;
        sys->register_event(fp, EventType::General, nullptr,
                            // chakra runtimes are in microseconds and we
                            // should convert it into nanoseconds
                            runtime);
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
        std::cout << "Generated broadcast DataSet, fp->my_id=" << fp->my_id << std::endl;
    } else {
        throw std::runtime_error("Unsupported collective comm type");
    }
    return true;
}

bool Workload::issue_send_comm(
    shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const auto src = node->comm_src<uint32_t>(this->sys->id);
    if (src != this->sys->id) {
        throw std::runtime_error("Send node should be issued by the sender");
    }
    const auto dst = node->comm_dst<uint32_t>();
    const auto size = node->comm_size<uint64_t>();
    // Record communication size for bandwidth calculation
    stats->get_operator_statistics(node->id()).comm_size = size;
    const auto tag = node->comm_tag<uint32_t>();

    //std::cout << "Send comm node: " << node->id() << " from " << src << " to " << dst << " size " << size << std::endl;


    // stg_tp2_pp2
    int cur_comm_group_id = -1;

    // BEGIN RECONFIGURATION LOGIC
    #ifndef CONGESTION_AWARE

    // 0 = PP dim (asymmetric send across stages); -1 -> 1 = DP dim
    // (same stage). all2allv EP sends can override this with reconfig_dim=3
    // so point-to-point MoE dispatch/combine uses the EP topology.
    int pp_prev = (static_cast<int>(dst) / AstraSim::Scheduler::num_npu_per_pp !=
                   this->sys->id / AstraSim::Scheduler::num_npu_per_pp) ? 0 : -1;
    bool skip_scheduler_reconfig = node->get_attr<int>("skip_scheduler_reconfig", 0) != 0;
    if (node->has_attr("reconfig_dim")) {
        pp_prev = node->get_attr<int>("reconfig_dim");
    }
    if (!skip_scheduler_reconfig) {
        bool reconfig_success = this->scheduler->pre_reconfig(dst, pp_prev);
        if (!reconfig_success) {
            timespec_t delta = {NS, 100000.0};
            this->sys->comm_NI->sim_schedule(delta, &Sys::handleEvent,
                new BasicEventHandlerData(this->sys->id, EventType::CallEvents));
            return false;
        }
    }

    // The decimal EP scheduler needs point-to-point transfers to hold the
    // reconfiguration gate. Legacy bitmask experiments retain their original
    // nonblocking send behavior for paper-result compatibility.
    if (AstraSim::Scheduler::decimal_topology_ids) {
        sys->increment_inflight_coll(this->sys->id,
                                     "SEND#" + std::to_string((int)node->id()));
    }

    #endif

    if (sys->trace_enabled) {
        std::cout << "RANK: " << this->sys->id << " at time " << Sys::boostedTick() <<" Issuing SEND " << src << "-" << dst << std::endl;
        std::cout << "RANK: " << this->sys->id << " inflight collective count: " << sys->get_inflight_coll() << std::endl;
    }
    previous_group_id = cur_comm_group_id;


    sim_request snd_req;
    snd_req.srcRank = src;
    snd_req.dstRank = dst;
    snd_req.reqType = UINT8;
    SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
    sehd->callable = this;
    sehd->wlhd = new WorkloadLayerHandlerData;
    sehd->wlhd->node_id = node->id();
    sehd->wlhd->src = src;
    sehd->wlhd->dst = dst;
    sehd->event = EventType::PacketSent;
    sys->front_end_sim_send(0, (void *)((long)dst), size, UINT8, dst, tag, &snd_req,
                            Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, sehd);
    return true;
}

bool Workload::issue_recv_comm(
    shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const auto src = node->comm_src<uint32_t>();
    const auto dst = node->comm_dst<uint32_t>(this->sys->id);
    if (dst != this->sys->id) {
        throw std::runtime_error("Recv node should be issued by the receiver");
    }
    const auto size = node->comm_size<uint64_t>();
    // Record communication size for bandwidth calculation
    stats->get_operator_statistics(node->id()).comm_size = size;
    const auto tag = node->comm_tag<uint32_t>();

    // stg_tp2_pp2 no need to reconfigure because recv is paired with send
    // if(previous_group_id >= 0) {
    //     // TODO use suitable topo_id
    //     int topo_id = 1;
    //     bool can_config = sys->comm_NI->sim_reconfig(topo_id);
    //     if (!can_config) return false;
    //     std::cout << "RANK: " << this->sys->id << " Switching to SEND/RECV group " << std::endl;
    // }

    // int cur_comm_group_id = -1;
    // bool reconfig_success = this->scheduler->pre_reconfig(cur_comm_group_id, previous_group_id);
    // if (!reconfig_success) return false;

    // previous_group_id = -1;

    // int cur_comm_group_id = -1;
    // bool reconfig_success = this->scheduler->pre_reconfig(cur_comm_group_id, previous_group_id);
    // if (!reconfig_success) return false;

    if (sys->trace_enabled) {
        std::cout << "RANK: " << this->sys->id << " at time " << Sys::boostedTick() <<" Issuing RECV " << src << "-" << dst << std::endl;
    }

    // sys->increment_inflight_coll(this->sys->id, std::to_string((int)node->id()) + " " + std::to_string(this->sys->id) + " RECV FROM " + std::to_string(src));

    sim_request rcv_req;
    RecvPacketEventHandlerData* rcehd = new RecvPacketEventHandlerData;
    rcehd->wlhd = new WorkloadLayerHandlerData;
    rcehd->wlhd->node_id = node->id();
    rcehd->workload = this;
    rcehd->event = EventType::PacketReceived;
    rcehd->wlhd->src = src;
    rcehd->wlhd->dst = dst;
    sys->front_end_sim_recv(0, (void *)((long)src), size, UINT8, src, tag, &rcv_req,
                            Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, rcehd);
    return true;
}

void Workload::skip_invalid(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    const auto node_id = node->id();
    auto& dependancy_resolver = this->et_feeder->getDependancyResolver();
    dependancy_resolver.finish_node(node_id);
    auto logger = LoggerFactory::get_logger("workload");
    logger->debug("callback,sys->id={}, tick={}, node->id={}, "
                  "node->name={}, node->type={}",
                  sys->id, Sys::boostedTick(), node->id(), node->name(),
                  static_cast<uint64_t>(node->type()));
    hw_resource->release(node);
    stats->record_end(node, Sys::boostedTick());
    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->recordEnd(node, Sys::boostedTick());
    }
}

void Workload::call(EventType event, CallData* data) {
    if (is_finished) {
        printf("Rank %d: workload already finished, ignore event %d\n", this->sys->id, static_cast<int>(event));
        return;
    }

    if (event == EventType::CollectiveCommunicationFinished) {
        TwoIntData* int_data = (TwoIntData*)data;
        uint64_t coll_comm_id = int_data->data;
        uint64_t comm_group_id = int_data->data2;
        uint64_t node_id = collective_comm_node_id_map[coll_comm_id];
        auto node = et_feeder->lookupNode(node_id);
        bool explicit_reconfig = node->has_attr("reconfig_topo_id");
        bool explicit_dim_reconfig = node->has_attr("reconfig_dim");
        bool skip_scheduler_reconfig =
            node->get_attr<int>("skip_scheduler_reconfig", 0) != 0;
        bool topology_agnostic_tp =
            is_topology_agnostic_tp_group(
                comm_groups[comm_group_id], comm_to_topo[comm_group_id],
                num_bw_matrix, explicit_reconfig || explicit_dim_reconfig);
        
        
        if (sys->trace_enabled) {
            printf("\033[0;32mRANK: %d at time %ld finish collective: %lu of comm group %lu, inflight collective %d\033[0m\n", this->sys->id, Sys::boostedTick(), coll_comm_id, comm_group_id, sys->get_inflight_coll());
        }

        #ifndef CONGESTION_AWARE

        if (topology_agnostic_tp || skip_scheduler_reconfig) {
            if (sys->trace_enabled && topology_agnostic_tp) {
                std::cout << "TP COMM pattern detected!" << std::endl;
            }
        } else {
            finish(comm_group_id);
            if (check_finish(comm_group_id)) {
                sys->decrement_inflight_coll(group_leader_npu_id[comm_group_id], -1);
                if (sys->trace_enabled) {
                    std::cout << "REMOVED inflight collective count for leader NPU " << group_leader_npu_id[comm_group_id] << ", current inflight collective count: " << sys->get_inflight_coll() << std::endl;
                }
                group_leader_npu_id.erase(comm_group_id);
                finish_rounds[comm_group_id] += 1;
                if (AstraSim::Scheduler::decimal_topology_ids) {
                    for (int npu_id : comm_groups[comm_group_id]->involved_NPUs) {
                        timespec_t delta = {NS, 0.0};
                        Sys::all_sys[npu_id]->comm_NI->sim_schedule(
                            delta, &Sys::handleEvent,
                            new BasicEventHandlerData(
                                npu_id, EventType::CallEvents));
                    }
                }
            }
        }

        #endif

        current_comm_group_idx++;
        // Legacy paper workloads use the provisioning look-ahead state machine.
        if (!AstraSim::Scheduler::decimal_topology_ids) {
            if (comm_to_topo[comm_group_id].size() == num_bw_matrix) {
                scheduler->post_reconfig(-2);
            } else {
                scheduler->post_reconfig(-1);
            }
        }
        //TODO edit coll_comm_id to comm group

        // if (current_comm_group_idx < comm_group_list.size()) {
        //     int next_comm_group_id = comm_group_list[current_comm_group_idx];
        //     int topo_id = 0;
        //     if (next_comm_group_id == 3 || next_comm_group_id == 4) {
        //         topo_id = 1;
        //     }
        //     bool can_config = sys->comm_NI->sim_reconfig(topo_id);
        //     printf("RANK: %d attempted to provision to topo_id: %d, result: %d\n", this->sys->id, topo_id, can_config);
        // }

        // if (coll_comm_id == 1921) {
        //     // TODO change coll_comm_id
        //     int topo_id = 0;
        //     bool can_config = sys->comm_NI->sim_reconfig(topo_id);
        //     printf("RANK: %d hard-code attempted to provision to topo_id: %d, result: %d\n", this->sys->id, topo_id, can_config);
        // }

        hw_resource->tics_gpu_comms += int_data->execution_time;

        if (sys->trace_enabled) {
            LoggerFactory::get_logger("workload")
                ->debug("callback,sys->id={}, tick={}, node->id={}, "
                        "node->name={}, node->type={}",
                        sys->id, Sys::boostedTick(), node->id(), node->name(),
                        static_cast<uint64_t>(node->type()));
        }

        hw_resource->release(node);
        stats->record_end(node, Sys::boostedTick());

        // Calculate network bandwidth
        auto& op_stat = stats->get_operator_statistics(node_id);
        Tick execution_time = int_data->execution_time;
        if (execution_time > 0 && op_stat.comm_size.has_value()) {
            double bandwidth =
                static_cast<double>(op_stat.comm_size.value()) / execution_time;
            op_stat.network_bandwidth = bandwidth;
        }

        if (this->sys->track_local_mem) {
            this->local_mem_usage_tracker->recordEnd(node, Sys::boostedTick());
        }

        this->et_feeder->getDependancyResolver().finish_node(node_id);

        issue_dep_free_nodes();

        // The Dataset class provides statistics that should be used later to
        // dump more statistics in the workload layer
        delete collective_comm_wrapper_map[coll_comm_id];
        collective_comm_wrapper_map.erase(coll_comm_id);

    } else {
        if (data == nullptr) {
            issue_dep_free_nodes();
        } else {
            WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
            shared_ptr<Chakra::FeederV3::ETFeederNode> node =
                et_feeder->lookupNode(wlhd->node_id);

            if (sys->trace_enabled) {
                LoggerFactory::get_logger("workload")
                    ->debug("callback,sys->id={}, tick={}, node->id={}, "
                            "node->name={}, node->type={}",
                            sys->id, Sys::boostedTick(), node->id(),
                            node->name(), static_cast<uint64_t>(node->type()));
            }

            hw_resource->release(node);
            stats->record_end(node, Sys::boostedTick());

            // Calculate network bandwidth for point-to-point communications
            if (event == EventType::PacketSent ||
                event == EventType::PacketReceived) {

                if (sys->trace_enabled) {
                    printf("\033[0;32mRANK: %d at time %ld finish SEND/RECV\033[0m\n", this->sys->id, Sys::boostedTick());
                }

                #ifndef CONGESTION_AWARE
                if (AstraSim::Scheduler::decimal_topology_ids) {
                    if (event == EventType::PacketSent) {
                        sys->decrement_inflight_coll(this->sys->id, node->id());
                    }
                } else if (event == EventType::PacketSent) {
                    scheduler->post_reconfig((long)wlhd->dst);
                } else if (event == EventType::PacketReceived) {
                    scheduler->post_reconfig((long)wlhd->src);
                }
                #endif

                auto& op_stat = stats->get_operator_statistics(wlhd->node_id);
                Tick execution_time =
                    stats->get_operator_statistics(wlhd->node_id).end_time -
                    stats->get_operator_statistics(wlhd->node_id).start_time;
                if (execution_time > 0 && op_stat.comm_size.has_value()) {
                    double bandwidth =
                        static_cast<double>(op_stat.comm_size.value()) /
                        execution_time;
                    op_stat.network_bandwidth = bandwidth;
                }
            }

            if (this->sys->track_local_mem) {
                this->local_mem_usage_tracker->recordEnd(node,
                                                         Sys::boostedTick());
            }

            this->et_feeder->getDependancyResolver().finish_node(wlhd->node_id);

            issue_dep_free_nodes();

            delete wlhd;
        }
    }

    const auto& dep_resolver = this->et_feeder->getDependancyResolver();
    if ((dep_resolver.get_dependancy_free_nodes().empty()) &&
        (dep_resolver.get_ongoing_nodes().empty()) &&
        (hw_resource->num_in_flight_cpu_ops == 0) &&
        (hw_resource->num_in_flight_gpu_comp_ops == 0) &&
        (hw_resource->num_in_flight_gpu_comm_ops == 0)) {
        report();
        sys->comm_NI->sim_notify_finished();
        is_finished = true;
    }
}

void Workload::fire() {
    call(EventType::General, NULL);
}

void Workload::report() {
    Tick curr_tick = Sys::boostedTick();
    LoggerFactory::get_logger("workload")
        ->info("sys[{}] finished, {} cycles, exposed communication {} cycles.",
               sys->id, curr_tick, curr_tick - hw_resource->tics_gpu_ops);
    stats->post_processing();
    stats->report();
    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->buildMemoryTrace();
        this->local_mem_usage_tracker->buildMemoryTimeline();
        this->local_mem_usage_tracker->dumpMemoryTrace(
            this->sys->local_mem_trace_filename);
        auto [peak_mem_usage, unit] =
            this->local_mem_usage_tracker->getPeakMemUsageFormatted();
        auto logger = LoggerFactory::get_logger("workload");
        logger->info("sys[{}] peak memory usage: {:.2f} {}", sys->id,
                     peak_mem_usage, unit);
        this->local_mem_usage_tracker.reset();
    }
}

AllToAllVLoadMapPtr Workload::load_all2allv_loads(const std::string& key) {
    static std::map<std::string, std::map<std::string, AllToAllVLoadMapPtr>> cache;

    auto file_it = cache.find(this->all2allv_load_filename);
    if (file_it == cache.end()) {
        ifstream inFile;
        inFile.open(this->all2allv_load_filename);
        if (!inFile.is_open()) {
            LoggerFactory::get_logger("workload")
                ->critical("Rank {} cannot open all2allv load cache {}",
                           sys->id, this->all2allv_load_filename);
            exit(EXIT_FAILURE);
        }

        json j;
        inFile >> j;
        std::map<std::string, AllToAllVLoadMapPtr> file_cache;
        for (json::iterator it = j.begin(); it != j.end(); ++it) {
            auto loads = std::make_shared<AllToAllVLoadMap>();
            for (const auto& entry : it.value().at("loads")) {
                int src = entry.at("src").get<int>();
                int dst = entry.at("dst").get<int>();
                uint64_t bytes = entry.at("bytes").get<uint64_t>();
                (*loads)[std::make_pair(src, dst)] = bytes;
            }
            file_cache[it.key()] = loads;
        }
        file_it = cache.emplace(this->all2allv_load_filename, file_cache).first;
    }

    auto key_it = file_it->second.find(key);
    if (key_it == file_it->second.end()) {
        LoggerFactory::get_logger("workload")
            ->critical("Rank {} cannot find all2allv load key {} in {}",
                       sys->id, key, this->all2allv_load_filename);
        exit(EXIT_FAILURE);
    }
    return key_it->second;
}

CommunicatorGroup* Workload::extract_comm_group(
    std::shared_ptr<Chakra::ETFeederNode> node) {
    std::string comm_group_name = node->pg_name<std::string>("");
    if (comm_group_name == "") {
        const int default_comm_group_id = -1;
        if (comm_groups.find(default_comm_group_id) == comm_groups.end()) {
            std::vector<int> involved_NPUs;
            for (int i = 0; i < sys->total_nodes; i++) {
                involved_NPUs.push_back(i);
            }
            comm_groups[default_comm_group_id] =
                new CommunicatorGroup(default_comm_group_id, involved_NPUs, sys);
        }
        return comm_groups[default_comm_group_id];
    }
    int comm_group_id = std::stoi(comm_group_name);
    if (comm_groups.find(comm_group_id) == comm_groups.end()) {
        LoggerFactory::get_logger("workload")
            ->critical(
                "For rank {} ET node {}, communicator group {} not found",
                sys->id, node->id(), comm_group_id);
        exit(EXIT_FAILURE);
    }
    return comm_groups[comm_group_id];
}
