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

#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;
using namespace AstraSim;
using namespace Chakra::FeederV3;
using json = nlohmann::json;

typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

std::map<int, int> Workload::group_vote_count = std::map<int, int>();
std::map<int, int> Workload::group_finish_count = std::map<int, int>();
std::map<int, int> Workload::group_leader_npu_id = std::map<int, int>();

std::map<int, int> Workload::vote_rounds = std::map<int, int>();
std::map<int, int> Workload::finish_rounds = std::map<int, int>();

std::map<int, std::map<int, int>> Workload::group_node_vote_count = std::map<int, std::map<int, int>>();
std::map<int, std::map<int, int>> Workload::group_node_finish_count = std::map<int, std::map<int, int>>();

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
        std::cout << "RANK: " << this->sys->id << " has already voted for comm group "
                 << comm_group_id << " in this round." << std::endl;
    }
    else{
        group_node_vote_count[comm_group_id][this->sys->id] += 1;
    }
    std::cout << "RANK: " << this->sys->id << " Vote for comm group "
              << comm_group_id << ", current node vote count: "
              << group_node_vote_count[comm_group_id][this->sys->id]
              << std::endl;
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
    std::cout << "RANK: " << this->sys->id << " Finish for comm group "
              << comm_group_id << ", current node finish count: "
              << group_node_finish_count[comm_group_id][this->sys->id]
              << std::endl;
}

bool Workload::check_vote(int comm_group_id) {
    int required_votes = comm_groups[comm_group_id]->involved_NPUs.size();
    int current_votes = 0;
    if (group_node_vote_count[comm_group_id][this->sys->id] - finish_rounds[comm_group_id] > 1) {
        std::cout << "RANK: " << this->sys->id << " is ahead, previous comm has not finished yet." << std::endl;
        return false;
    }


    std::cout << "RANK: " << this->sys->id << " checking votes for comm group " << comm_group_id << ": ";
    if (group_node_vote_count.find(comm_group_id) !=
        group_node_vote_count.end()) {
        for (const auto& pair : group_node_vote_count[comm_group_id]) {
            if (pair.second > finish_rounds[comm_group_id]) {
                current_votes += 1;
                std::cout << pair.first << " ";
            }
        }
    }

    std::cout << " -- (" << current_votes << "/" << required_votes << ") [" << finish_rounds[comm_group_id] << "]\n";
    return current_votes >= required_votes;
}

bool Workload::check_finish(int comm_group_id) {
    int required_finishes = comm_groups[comm_group_id]->involved_NPUs.size();
    int current_finishes = 0;
    std::cout << "RANK: " << this->sys->id << " checking finishes for comm group " << comm_group_id << ": ";
    if (group_node_finish_count.find(comm_group_id) !=
        group_node_finish_count.end()) {
        for (const auto& pair : group_node_finish_count[comm_group_id]) {
            if (pair.second > finish_rounds[comm_group_id]) {
                current_finishes += 1;
                std::cout << pair.first << " ";
            }
        }
    }

    std::cout << " -- (" << current_finishes << "/" << required_finishes << ") [" << finish_rounds[comm_group_id] << "]\n";
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
    initialize_comm_groups(comm_group_filename);
    this->stats = new Statistics(this);
    this->is_finished = false;

    this->scheduler = new Scheduler(sys, provision_config);
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
    
    std::cout << "Involved npus: ";
    for (auto d : comm_group->involved_NPUs) {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    // std::cout << std::endl;

    // TODO in addition to comm group detection, also check topo id change

    int comm_id = comm_group->get_id();


    std::cout << "\033[0;32mRANK: " << this->sys->id << " at time " << Sys::boostedTick() << " Issuing collective " << comm_group->to_string() << "\033[0m" << std::endl;
    std::cout << " COLL COMM NODE: " << node->id() << " of comm group " << comm_group->get_id() << std::endl;
    
    previous_group_id = comm_id;

    std::cout << "\033[0;32mRANK: " << this->sys->id << " inflight collective count: " << sys->get_inflight_coll() << "\033[0m" << std::endl;

    const auto comm_type =
        static_cast<ChakraCollectiveCommType>(node->comm_type<uint64_t>());
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
        std::cout << "Generated all-reduce DataSet, fp->my_id=" << fp->my_id << std::endl;
    } else if (comm_type == ChakraCollectiveCommType::ALL_TO_ALL) {
        DataSet* fp = sys->generate_all_to_all(comm_size, involved_dims,
                                               comm_group, comm_priority);
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


    std::cout << "RANK: " << this->sys->id << " at time " << Sys::boostedTick() <<" Issuing SEND " << src << "-" << dst << std::endl;
    previous_group_id = cur_comm_group_id;

    sim_request snd_req;
    snd_req.srcRank = src;
    snd_req.dstRank = dst;
    snd_req.reqType = UINT8;
    SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
    sehd->callable = this;
    sehd->wlhd = new WorkloadLayerHandlerData;
    sehd->wlhd->node_id = node->id();
    sehd->event = EventType::PacketSent;
    sys->front_end_sim_send(0, Sys::dummy_data, size, UINT8, dst, tag, &snd_req,
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


    std::cout << "RANK: " << this->sys->id << " at time " << Sys::boostedTick() <<" Issuing RECV " << src << "-" << dst << std::endl;

    sim_request rcv_req;
    RecvPacketEventHandlerData* rcehd = new RecvPacketEventHandlerData;
    rcehd->wlhd = new WorkloadLayerHandlerData;
    rcehd->wlhd->node_id = node->id();
    rcehd->workload = this;
    rcehd->event = EventType::PacketReceived;
    sys->front_end_sim_recv(0, Sys::dummy_data, size, UINT8, src, tag, &rcv_req,
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
        
        
        printf("\033[0;32mRANK: %d at time %ld finish collective: %lu of comm group %lu, inflight collective %d\033[0m\n", this->sys->id, Sys::boostedTick(), coll_comm_id, comm_group_id, sys->get_inflight_coll());

        current_comm_group_idx++;

        hw_resource->tics_gpu_comms += int_data->execution_time;
        uint64_t node_id = collective_comm_node_id_map[coll_comm_id];
        shared_ptr<Chakra::FeederV3::ETFeederNode> node =
            et_feeder->lookupNode(node_id);

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

                printf("\033[0;32mRANK: %d at time %ld finish SEND/RECV\033[0m\n", this->sys->id, Sys::boostedTick());

                // if(event == EventType::PacketSent)
                //     sys->decrement_inflight_coll(this->sys->id, node->id());
                

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

CommunicatorGroup* Workload::extract_comm_group(
    std::shared_ptr<Chakra::ETFeederNode> node) {
    std::string comm_group_name = node->pg_name<std::string>("");
    if (comm_group_name == "") {
        // No communicator group is specified for this communication ET node.
        return nullptr;
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
