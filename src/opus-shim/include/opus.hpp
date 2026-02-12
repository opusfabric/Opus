#pragma once

#include <torch/python.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/chrono.h>
#include <nccl.h>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <thread>
#include <queue>

#define PP_INTERVAL 1000000
#define PP_BACKEND_INTERVAL 10

namespace c10d {

static std::atomic<int> backend_count;

class BackendOpus : public Backend {
private:
    int backendId_;
    ncclComm_t comm_;
    int rank_;
    int nodeRank_;
    int localRank_;
    int size_;
    cudaStream_t stream_;
    ncclUniqueId id_;
    std::string  hexId_;

    int ctrlSock_;

    int collectiveCounter_; // pre
    int collectiveCounter_post_;
    int collectiveCounter_main_;

    // For PP send/recv. The counter space will be partitioned by the sender and recver rank.
    int up_cnt_;
    int down_cnt_;
    int up_cnt_post_;
    int down_cnt_post_;
    int up_cnt_main_;
    int down_cnt_main_;
    int get_pp_base(int src_rank, int dst_rank, int cur_rank) {return std::max(std::max(src_rank, dst_rank), cur_rank);}; 
    // max(src_rank, dst_rank) * 1000000 + cnt

    bool is_pp_backend_ = false;

    int ncclActiveGroupCounter_ = 0;

    int domain_ = 0; // 0 is scale-out (hybrid), 1 is scale-up
    int network_type = 0; // 0 is cpu frontend, 1 is gpu backend
    
    // shared among all BackendOpus instances
    int mode = 0; // 0: baseline, 1: provision, 2: no-ctl
    int is_emulation = 0; // 0: real CTL, 1: emulated
    int use_cuda_stream_ = 0; // 0 pre_ post both use host threads, 1 use cuda for pre, 2 use cuda for post, 3 use cuda for both, only used for provisioning
    float emulated_latency_ms = 0.0; // delay in ms for emulation



    struct CommStage {
        int backendId_start;
        int collCounter_start;
        int backendId_end;
        int collCounter_end;
    };

    std::vector<CommStage> comm_stages_;

    static std::vector<std::string> backendId_hexId_map;
    static bool topo_ready;
    static int curCommStageIdx_;
    static int curCommStageIdx_post_;
    static std::mutex topo_mutex_;
    static std::condition_variable topo_cv_;
    static int pp_stage_;

    bool comm_change_after_ = false;
    bool comm_change_before_ = false;
    
    // pre_coll post coll threading
    std::thread preThread_;
    std::thread postThread_;
    std::atomic<bool> running_{true};
    
    std::queue<std::function<void()>> preQueue_;
    std::mutex preMutex_;
    std::condition_variable preCV_;
    
    std::queue<std::function<void()>> postQueue_;
    std::mutex postMutex_;
    std::condition_variable postCV_;
    
    void preThreadFunc();
    void postThreadFunc();

    void checkin();
    void populate_comm_stages();
    void pre_coll(std::string coll_type = "", int input_size = 0, int output_size = 0, int src_rank = -1, int dst_rank = -1);

    bool is_in_comm_stage();
    bool is_comm_change_before(int collectiveCounter, int curCommStageIdx);
    bool is_comm_change_after(int collectiveCounter, int curCommStageIdx);
    bool check_comm_stage();

    void post_coll(std::string coll_type, int src_rank = -1, int dst_rank = -1);

    void groupStart();
    void groupEnd();

    std::vector<std::string> SERVER_IPS = 
    {
        "128.253.51.206",
        "128.253.51.207",
        "128.253.51.208",
        "128.253.51.209"
    };

public:

    BackendOpus(int rank, int size, c10::intrusive_ptr<Options> options = nullptr);
    ~BackendOpus();

    int getMode() const {
        return mode;
    }

    c10::intrusive_ptr<Work> broadcast(
        std::vector<at::Tensor>& data,
        const BroadcastOptions& opts = BroadcastOptions()) override;

    c10::intrusive_ptr<Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const AllreduceOptions& opts = AllreduceOptions()) override;

    c10::intrusive_ptr<Work> allreduce_coalesced(
        std::vector<at::Tensor>& tensors,
        const AllreduceCoalescedOptions& opts =
            AllreduceCoalescedOptions()) override;

    c10::intrusive_ptr<Work> reduce(
        std::vector<at::Tensor>& tensors,
        const ReduceOptions& opts = ReduceOptions()) override;

    c10::intrusive_ptr<Work> allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> _allgather_base(
        at::Tensor& outputBuffer,
        at::Tensor& inputBuffer,
        const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
        std::vector<at::Tensor>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> barrier(
        const BarrierOptions& opts = BarrierOptions()) override;

    c10::intrusive_ptr<Work> gather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const GatherOptions& opts = GatherOptions()) override;

    c10::intrusive_ptr<Work> scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const ScatterOptions& opts = ScatterOptions()) override;

    c10::intrusive_ptr<Work> reduce_scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

    c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
        std::vector<at::Tensor>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

    c10::intrusive_ptr<Work> _reduce_scatter_base(
        at::Tensor& outputBuffer,
        at::Tensor& inputBuffer,
        const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

    c10::intrusive_ptr<Work> alltoall_base(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        std::vector<int64_t>& outputSplitSizes,
        std::vector<int64_t>& inputSplitSizes,
        const AllToAllOptions& opts = AllToAllOptions()) override;

    c10::intrusive_ptr<Work> alltoall(
        std::vector<at::Tensor>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllToAllOptions& opts = AllToAllOptions()) override;

    c10::intrusive_ptr<Work> send(
        std::vector<at::Tensor>& tensors,
        int dstRank,
        int tag) override;

    c10::intrusive_ptr<Work> recv(
        std::vector<at::Tensor>& tensors,
        int srcRank,
        int tag) override;

    c10::intrusive_ptr<Work> recvAnysource(
        std::vector<at::Tensor>& tensors,
        int tag) override;

    static c10::intrusive_ptr<Backend> createBackendOpus(
        const c10::intrusive_ptr<::c10d::Store>& store,
        int rank,
        int size,
        const std::chrono::duration<float>& timeout);

    static void BackendOpusConstructor() __attribute__((constructor)) {
        py::object module = py::module::import("torch.distributed");
        py::object register_backend =
            module.attr("Backend").attr("register_backend");
        register_backend("Opus", py::cpp_function(createBackendOpus));
    }

    int getRank() const {
        return rank_;
    }

    int getBackendId() const {
        return backendId_;
    }

    int getCollectiveCounter() const {
        return collectiveCounter_;
    }

    bool supportsCoalescing() const override {
        return true;
    }

    const std::string getBackendName() const override {
        return std::string("opus");
    }

    void startCoalescing() override;
    c10::intrusive_ptr<Work> endCoalescing() override;
    

    void topowrite(int cnt);
    void topoprovision();
    
    void signalTopoReady() {
        std::lock_guard<std::mutex> lk(topo_mutex_);
        topo_ready = true;
        topo_cv_.notify_all();
    }
    
    void pre_coll_callback_helper(int src_rank = -1, int dst_rank = -1);
    void post_coll_callback_helper(int src_rank = -1, int dst_rank = -1);
};

class WorkOpus : public Work {
    friend class BackendOpus;
public:
    WorkOpus(
        OpType opType,
        c10::intrusive_ptr<c10::ivalue::Future> future,
        cudaStream_t stream) // future of the output
        : Work(
              -1, // rank, only used by recvAnySource, irrelevant in this demo
              opType),
          future_(std::move(future)),
          stream_(stream) {
            cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
            cudaEventRecord(event_, stream_);
          }
    ~WorkOpus() override {
        cudaEventDestroy(event_);
    }
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

private:
    c10::intrusive_ptr<c10::ivalue::Future> future_;
    cudaStream_t stream_;
    cudaEvent_t event_;
};

// instantiating static variables
bool BackendOpus::topo_ready = true;
int BackendOpus::pp_stage_ = 0;
int BackendOpus::curCommStageIdx_ = 0;
int BackendOpus::curCommStageIdx_post_ = 0;
std::vector<std::string> BackendOpus::backendId_hexId_map = std::vector<std::string>(7, "");

std::mutex BackendOpus::topo_mutex_;
std::condition_variable BackendOpus::topo_cv_;

} // namespace c10d
