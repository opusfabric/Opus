#include <iostream>
#include <thread>
#include <sstream>
#include <vector>
#include <fstream>
#include <toml++/toml.h> // Include the toml++ library header


void customPrint(const std::string& message, int rank, int backendId = -1) {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_tm = *std::localtime(&now_time_t);

    auto duration = now.time_since_epoch();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() % 1000000;

    std::thread::id this_id = std::this_thread::get_id();
    std::ostringstream oss;
    oss << this_id;
    std::string thread_id_str = oss.str();

    static std::hash<int> hasher;
    size_t hash_value = hasher(rank);
    int color_code = 31 + (hash_value % 7); // ANSI color codes (31-37 for text colors)
    printf("\033[%dmBackend %d RANK %d tid: %s t:[%02d:%02d:%02d.%06ld]: %s\033[0m\n", 
           color_code, backendId, rank, thread_id_str.c_str(), now_tm.tm_hour, now_tm.tm_min, 
           now_tm.tm_sec, microseconds, message.c_str());
}

bool is_pp_enabled() {
    int pipeline_parallel_degree = 1;
    const char* config_file = std::getenv("CONFIG_FILE");
    if (config_file == nullptr) {
        throw std::runtime_error("CONFIG_FILE environment variable is not set.");
    }

    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open configuration file.");
    }

    toml::table config = toml::parse_file(config_file);

    auto* parallelism = config["parallelism"].as_table();
    if (!parallelism) {
        return false;
    }

    pipeline_parallel_degree =
        parallelism->get("pipeline_parallel_degree")->value_or(1);
    return pipeline_parallel_degree > 1;
}

bool is_1d_parallelism() {
    int data_parallel_replicate_degree = 1;
    int data_parallel_shard_degree = 1;
    int tensor_parallel_degree = 1;
    int pipeline_parallel_degree = 1;
    const char* config_file = std::getenv("CONFIG_FILE");
    if (config_file == nullptr) {
        throw std::runtime_error("CONFIG_FILE environment variable is not set.");
    }

    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open configuration file.");
    }

    toml::table config = toml::parse_file(config_file);

    auto* parallelism = config["parallelism"].as_table();
    if (!parallelism) {
        throw std::runtime_error("Missing [parallelism] section in config file.");
    }

    data_parallel_replicate_degree =
        parallelism->get("data_parallel_replicate_degree")->value_or(1);

    data_parallel_shard_degree =
        parallelism->get("data_parallel_shard_degree")->value_or(1);

    tensor_parallel_degree =
        parallelism->get("tensor_parallel_degree")->value_or(1);

    pipeline_parallel_degree =
        parallelism->get("pipeline_parallel_degree")->value_or(1);

    int degrees_greater_than_one = 0;
    if (data_parallel_replicate_degree > 1) degrees_greater_than_one++;
    if (data_parallel_shard_degree > 1) degrees_greater_than_one++;
    if (tensor_parallel_degree > 1) degrees_greater_than_one++;
    if (pipeline_parallel_degree > 1) degrees_greater_than_one++;

    return degrees_greater_than_one == 1;
}

void determine_server_addr(std::string& server_ip, int& port, int global_rank, int local_rank, int backendId_, const std::vector<std::string>SERVER_IPS) {
    // modify server_ip and port

    int data_parallel_replicate_degree = 1;
    int data_parallel_shard_degree = 1;
    int tensor_parallel_degree = 1;
    int pipeline_parallel_degree = 1;
    const char* config_file = std::getenv("CONFIG_FILE");
    if (config_file == nullptr) {
        throw std::runtime_error("CONFIG_FILE environment variable is not set.");
    }

    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open configuration file.");
    }

    toml::table config = toml::parse_file(config_file);

    auto* parallelism = config["parallelism"].as_table();
    if (!parallelism) {
        throw std::runtime_error("Missing [parallelism] section in config file.");
    }

    data_parallel_replicate_degree =
        parallelism->get("data_parallel_replicate_degree")->value_or(1);

    data_parallel_shard_degree =
        parallelism->get("data_parallel_shard_degree")->value_or(1);

    tensor_parallel_degree =
        parallelism->get("tensor_parallel_degree")->value_or(1);

    pipeline_parallel_degree =
        parallelism->get("pipeline_parallel_degree")->value_or(1);

    const char* num_nodes_env = std::getenv("NUM_NODES");
    const char* num_ranks_per_node_env = std::getenv("NUM_RANKS_PER_NODE");

    if (num_nodes_env == nullptr || num_ranks_per_node_env == nullptr) {
        throw std::runtime_error("NUM_NODES or NUM_RANKS_PER_NODE environment variable is not set.");
    }

    int num_nodes = std::stoi(num_nodes_env);
    int num_ranks_per_node = std::stoi(num_ranks_per_node_env);

    int parallelism_product =
        data_parallel_replicate_degree * data_parallel_shard_degree *
        tensor_parallel_degree * pipeline_parallel_degree;
    if (num_nodes * num_ranks_per_node != parallelism_product) {
        std::cerr << "Total ranks: " << num_nodes * num_ranks_per_node
              << ", Parallelism degrees product: " << parallelism_product
              << std::endl;
        throw std::runtime_error("Mismatch between total ranks and parallelism degrees.");
    }

    int total_num_parallelism = (data_parallel_replicate_degree > 1 ? 1 : 0)
        + (data_parallel_shard_degree > 1 ? 1 : 0)
        + (tensor_parallel_degree > 1 ? 1 : 0)
        + (pipeline_parallel_degree > 1 ? 1 : 0);

    server_ip = SERVER_IPS[0];
    port = 34567;

    std::cout << "Total parallelism: " << total_num_parallelism << std::endl;

    if (total_num_parallelism == 3) {
        // backendId 0, 1, 2, 3 TODO
        int inner_p_degree = tensor_parallel_degree;
        int mid_p_degree = data_parallel_shard_degree * tensor_parallel_degree;
        int outer_p_degree = pipeline_parallel_degree * data_parallel_shard_degree * tensor_parallel_degree;

        if (backendId_ == 3) {
            // contiguous, tp
            int server_global_rank = global_rank / inner_p_degree * inner_p_degree;
            server_ip = SERVER_IPS[server_global_rank / num_ranks_per_node];
            port = port + 1 + global_rank / inner_p_degree;

        } else if (backendId_ == 2) {
            // dp
            int server_global_rank = global_rank / mid_p_degree * mid_p_degree + global_rank % tensor_parallel_degree; 
            server_ip = SERVER_IPS[server_global_rank / num_ranks_per_node];
            port = port + 1 + outer_p_degree / tensor_parallel_degree + global_rank / mid_p_degree * tensor_parallel_degree + global_rank % tensor_parallel_degree + 50;

        } else if (backendId_ == 1) {
            // pp
            int server_global_rank = global_rank % mid_p_degree;
            server_ip = SERVER_IPS[server_global_rank / num_ranks_per_node];
            port = port + 1 + outer_p_degree / tensor_parallel_degree + outer_p_degree / data_parallel_shard_degree + global_rank % mid_p_degree + 50 * 2;

        } else if (backendId_ != 0){
            // TODO assume same as 2 for 4, 5, and 6
            // TODO verify correctness

            // int server_global_rank = global_rank / mid_p_degree * mid_p_degree + global_rank % tensor_parallel_degree; 
            // server_ip = SERVER_IPS[server_global_rank / num_ranks_per_node];
            // port = port + 1 + outer_p_degree / tensor_parallel_degree + global_rank / mid_p_degree * tensor_parallel_degree + global_rank % tensor_parallel_degree;

            int server_global_rank = global_rank / mid_p_degree * mid_p_degree + global_rank % tensor_parallel_degree; 
            server_ip = SERVER_IPS[server_global_rank / num_ranks_per_node];

            port = port + 1 + outer_p_degree / tensor_parallel_degree + mid_p_degree + outer_p_degree / data_parallel_shard_degree * (backendId_ - 3) + global_rank / mid_p_degree * tensor_parallel_degree + global_rank % tensor_parallel_degree + 50 * (backendId_ - 1);
        }

    } else if (total_num_parallelism == 2) {
        // backendId 0, 1, 2
        int inner_parallelism_degree = tensor_parallel_degree > 1 ? tensor_parallel_degree : 
            data_parallel_shard_degree > 1 ? data_parallel_shard_degree : 1;

        if (inner_parallelism_degree == 1) {
            throw std::runtime_error("Inner parallelism degree cannot be 1 when total parallelism is 2.");
        }

        int outer_parallelism_degree = pipeline_parallel_degree * tensor_parallel_degree * data_parallel_shard_degree / inner_parallelism_degree;
        

        if (backendId_ == 2) {
            // contiguous
            int server_global_rank = global_rank / inner_parallelism_degree * inner_parallelism_degree;
            server_ip = SERVER_IPS[server_global_rank / num_ranks_per_node];
            port = port + 1 + global_rank / inner_parallelism_degree;

        } else if (backendId_ == 1) {
            int server_global_rank = global_rank % inner_parallelism_degree;
            server_ip = SERVER_IPS[server_global_rank / num_ranks_per_node];
            port = port + 1 + outer_parallelism_degree + server_global_rank;
        }  else if (backendId_ != 0){
            // TODO verify correctness
            if (tensor_parallel_degree > 1) {
                // same as 1 for 3 and 4
                int server_global_rank = global_rank % inner_parallelism_degree;
                server_ip = SERVER_IPS[server_global_rank / num_ranks_per_node];

                port = port + 1 + outer_parallelism_degree + inner_parallelism_degree * (backendId_ - 2) + server_global_rank;
            } else {
                // same as 2
                int server_global_rank = global_rank / inner_parallelism_degree * inner_parallelism_degree;
                server_ip = SERVER_IPS[server_global_rank / num_ranks_per_node];

                port = port + 1 + outer_parallelism_degree * (backendId_ - 2) + inner_parallelism_degree + global_rank / inner_parallelism_degree;
            }
            
        }
    }
}
