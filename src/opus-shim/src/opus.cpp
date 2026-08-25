#include "NCCLUtils.hpp"
#include "opus.hpp"
#include <iostream>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <torch/torch.h> 
#include <torch/extension.h>
#include <cuda.h>
#include <string>
#include <sstream>
#include <ATen/record_function.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
// #include <nvtx3/nvtx3.hpp> 
#include <netinet/tcp.h>
#include "util.hpp"


namespace net {

inline ssize_t send(int sockfd, const void* buf, size_t len, int flags) {
    return ::send(sockfd, buf, len, flags);
}

inline ssize_t recv(int sockfd, void* buf, size_t len, int flags) {
    return ::recv(sockfd, buf, len, flags);
}

}

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

namespace opus_utils {

// NCCL type typing
static std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
    {at::kBool, ncclUint8},
#ifdef NCCL_SUPPORTS_FP8
    {at::kFloat8_e5m2, ncclFloat8e5m2},
    {at::kFloat8_e4m3fn, ncclFloat8e4m3},
#else
    {at::kFloat8_e5m2, ncclUint8},
    {at::kFloat8_e4m3fn, ncclUint8},
#endif
    {at::kFloat8_e4m3fnuz, ncclUint8},
    {at::kFloat8_e5m2fnuz, ncclUint8},
#if HAS_NCCL_BF16_DATATYPE
    {at::kBFloat16, ncclBfloat16},
#endif // HAS_NCCL_BF16_DATATYPE
};

ncclDataType_t getNcclDataType(at::ScalarType type) {
    auto it = ncclDataType.find(type);
    if (it == ncclDataType.end()) {
      throw std::runtime_error(
        "Input tensor data type is not supported for NCCL process group: ");
    }
    return it->second;
}

} // namespace opus_utils

static uint64_t getHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash(const char* hostname) {
  char hostHash[1024];

  // Fall back is the hostname if something fails
  (void) strncpy(hostHash, hostname, sizeof(hostHash));
  int offset = strlen(hostHash);

  FILE *file = fopen(HOSTID_FILE, "r");
  if (file != NULL) {
    char *p;
    if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        free(p);
    }
  }
  fclose(file);

  // Make sure the string is terminated
  hostHash[sizeof(hostHash)-1]='\0';

  return getHash(hostHash);
}



static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

void setSocketReuse(int sockfd) {
    int opt = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("Error setting SO_REUSEADDR");
        exit(EXIT_FAILURE);
    }
}

// Function to establish a server socket (TCP)
int createServerSocket(const char* server_ip, int port) {
  int sockfd;
  struct sockaddr_in server_addr;

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    perror("Error creating socket");
    exit(EXIT_FAILURE);
  }

  // Get server IP
  struct addrinfo hints{}, *res;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  int ret = getaddrinfo(server_ip, NULL, &hints, &res);
  if (ret != 0) {
      fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(ret));
      exit(EXIT_FAILURE);
  }

  setSocketReuse(sockfd);
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  server_addr.sin_addr = ((struct sockaddr_in*)res->ai_addr)->sin_addr;

  freeaddrinfo(res);

  if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
    perror("Error binding socket");
    exit(EXIT_FAILURE);
  }

  if (listen(sockfd, 100) < 0) {
    perror("Error listening on socket");
    exit(EXIT_FAILURE);
  }

  return sockfd;
}

// Function to establish a client socket (TCP)
int createClientSocket(const char *server_ip, int port) {
    int sockfd;
    struct sockaddr_in server_addr;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        return -1;
    }

    struct addrinfo hints{}, *res = NULL;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    int ret = getaddrinfo(server_ip, NULL, &hints, &res);
    if (ret != 0 || res == NULL) {
        return -1;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr = ((struct sockaddr_in*)res->ai_addr)->sin_addr;

    freeaddrinfo(res);

    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        // perror("Connection failed");
        // exit(EXIT_FAILURE);
        return -1;
    }

    return sockfd;
}

namespace c10d {


bool WorkOpus::isCompleted() {
  cudaError_t status = cudaEventQuery(event_);
  return (status == cudaSuccess);
}

bool WorkOpus::isSuccess() const {
  cudaError_t status = cudaEventQuery(event_);
  return (status == cudaSuccess);
}

bool WorkOpus::wait(std::chrono::milliseconds timeout) {
  auto start = std::chrono::steady_clock::now();
  while (cudaEventQuery(event_) == cudaErrorNotReady) {
      if (std::chrono::steady_clock::now() - start > timeout) {
          return false; // timeout
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  if (future_ && !future_->completed()) {
      future_->markCompleted(c10::IValue());
  }
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkOpus::getFuture() {
  return future_;
}

static std::string bytesToHex(const char* data, size_t len) {
  static const char* hex = "0123456789ABCDEF";
  std::string s; s.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    unsigned char c = data[i];
    s.push_back(hex[c>>4]);
    s.push_back(hex[c & 0xF]);
  }
  return s;
}

static int persistentCtlSock = -1;
static bool persistentCtlSockInitialized = false;

int getOrInitCtlSock() {
  if (!persistentCtlSockInitialized) {
    struct sockaddr_in controller_addr;
    int port = 1234;
    if (const char* port_env = std::getenv("OPUS_CONTROLLER_PORT")) {
      try {
        port = std::stoi(port_env);
      } catch (const std::exception&) {
        printf("[CTL Init] Invalid OPUS_CONTROLLER_PORT; using 1234\n");
        port = 1234;
      }
    }
    if (port < 1 || port > 65535) {
      printf("[CTL Init] OPUS_CONTROLLER_PORT must be between 1 and 65535\n");
      return -1;
    }

    persistentCtlSock = socket(AF_INET, SOCK_STREAM, 0);
    if (persistentCtlSock < 0) {
      printf("[CTL Init] Failed to construct socket\n");
      return -1;
    }

    int flag = 1;
    setsockopt(persistentCtlSock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    struct timeval timeout{0, 200 * 1000}; // set timeout to be 200ms
    setsockopt(persistentCtlSock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    struct addrinfo hints{}, *res = NULL;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    // Read controller IP from environment variable CONTROLLER_IP
    const char* controller_ip = std::getenv("CONTROLLER_IP");
    if (controller_ip == nullptr) {
      printf("[CTL Init] CONTROLLER_IP environment variable not set\n");
      return -1;
    }
    int ret = getaddrinfo(controller_ip, NULL, &hints, &res);
    if (ret != 0 || res == NULL) {
        return -1;
    }

    memset(&controller_addr, 0, sizeof(controller_addr));
    controller_addr.sin_family = AF_INET;
    controller_addr.sin_port = htons(port);
    controller_addr.sin_addr = ((struct sockaddr_in*)res->ai_addr)->sin_addr;

    freeaddrinfo(res);

    if (connect(persistentCtlSock, (sockaddr*)&controller_addr, sizeof(controller_addr)) != 0) {
      printf("[CTL Init] Failed to connect to controller\n");
      close(persistentCtlSock);
      persistentCtlSock = -1;
      return -1;
    }

    persistentCtlSockInitialized = true;
    printf("[CTL INIT] Connected to controller on socket %d\n", persistentCtlSock);
  }

  return persistentCtlSock;
}

void BackendOpus::checkin() {
  // send hexId and nranks to controller
  hexId_ = bytesToHex(reinterpret_cast<const char*>(&id_), 32);
  backendId_hexId_map[backendId_] = hexId_;

  int s = getOrInitCtlSock();
  if (s >= 0) {
    std::ostringstream oss;
    oss << "COMM;" << hexId_ << "," << size_ << "," << localRank_ << "\n";
    std::string msg = oss.str();
    net::send(s, msg.data(), msg.size(), 0);
    // printf("[SENT TO CTL] %s\n", msg.c_str());
    ctrlSock_ = s;
    customPrint("[CTL] Sent CHECKIN, hex:" + hexId_, rank_, backendId_);
  } else {
    customPrint("[CTL] Failed to connect to controller", rank_, backendId_);
    close(s);
    exit(0);
  }

  // receive ack
  char buf[512];
  std::string accum;
 
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000); // total wait

  while (std::chrono::steady_clock::now() < deadline) {

      ssize_t n = net::recv(ctrlSock_, buf, sizeof(buf), 0);

      if (n > 0) {
          accum.append(buf, n);
      }
      else if (n == 0) {
          customPrint("[CTL] Controller closed connection", rank_, backendId_);
          return;
      }
      else { // n < 0
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
              // 200ms timeout expired → retry
              continue;
          } else {
              perror("recv");
              customPrint("[CTL] recv() failed", rank_, backendId_);
              return;
          }
      }

      // process all complete lines
      size_t pos;
      while ((pos = accum.find('\n')) != std::string::npos) {
          std::string line = accum.substr(0, pos);
          accum.erase(0, pos + 1);

          auto sep = line.find(';');
          std::string hdr  = sep == std::string::npos ? line : line.substr(0, sep);
          std::string body = sep == std::string::npos ? ""   : line.substr(sep + 1);

          if (hdr == "CHECKIN_ACK") {
              auto body_sep = body.find(',');
              std::string receivedHexId = body.substr(0, body_sep);
              std::string domainStr = body.substr(body_sep + 1);

              if (receivedHexId == hexId_) {
                  domain_ = std::stoi(domainStr);
                  if (domain_ >= 0) {
                      customPrint("[CTL] Received CHECKIN_ACK, domain=" + std::to_string(domain_), rank_, backendId_);
                      return;
                  } else {
                      customPrint("[CTL] Invalid domain in CHECKIN_ACK", rank_, backendId_);
                      return;
                  }
              } else {
                  customPrint("[CTL] CHECKIN_ACK hexId mismatch", rank_, backendId_);
                  return;
              }
          }
      }
  }

  customPrint("[CTL] CHECKIN_ACK timeout", rank_, backendId_);
}

void BackendOpus::topoprovision() {
  // Send CONFIG_REQ
  if (is_emulation == 1) {
    // Emulate latency
    std::ostringstream oss;
    oss << "[CTL] Emulating provisioning latency of " << emulated_latency_ms << " ms";
    customPrint(oss.str(), rank_, backendId_);
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(emulated_latency_ms)));
    return;
  }

  if (curCommStageIdx_post_ >= comm_stages_.size()) {
    customPrint("[CTL] No more comm stages to provision", rank_, backendId_);
    return;
  }
 
  int next_backendId = comm_stages_[curCommStageIdx_post_].backendId_start;
  int next_collCounter = comm_stages_[curCommStageIdx_post_].collCounter_start;
  std::string new_hexId = backendId_hexId_map[next_backendId];

  if (!is_pp_backend_) {
    next_collCounter += pp_stage_ * PP_INTERVAL;
  }

  std::string msg = "CONFIG_REQ;" + new_hexId + "," + std::to_string(next_backendId) 
    + "," + std::to_string(next_collCounter) + "\n";

  net::send(ctrlSock_, msg.data(), msg.size(), 0);
  // printf("[SENT TO CTL] %s\n", msg.c_str());

  std::ostringstream oss;
  oss << "PROVISION topo for next backend: " << next_backendId << ", Idx: " << next_collCounter;
  customPrint(oss.str(), rank_, backendId_);

  // Wait for CONFIG_ACK
  char buf[512];
  std::string accum;
  // printf("[DEBUG] Waiting for CONFIG_ACK on socket %d\n", ctrlSock_);
  
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(10000); // total wait

  while (std::chrono::steady_clock::now() < deadline) {

      ssize_t n = net::recv(ctrlSock_, buf, sizeof(buf), 0);

      if (n > 0) {
          accum.append(buf, n);
      }
      else if (n == 0) {
          customPrint("[CTL] Controller closed connection", rank_, backendId_);
          return;
      }
      else { // n < 0
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
              // timeout slice expired → retry
              continue;
          } else {
              perror("recv");
              customPrint("[CTL] recv failed", rank_, backendId_);
              return;
          }
      }

      // process all complete lines
      size_t pos;
      while ((pos = accum.find('\n')) != std::string::npos) {
          std::string line = accum.substr(0, pos);
          accum.erase(0, pos + 1);

          auto sep = line.find(';');
          std::string hdr  = sep == std::string::npos ? line : line.substr(0, sep);
          std::string body = sep == std::string::npos ? ""   : line.substr(sep + 1);

          if (hdr == "CONFIG_ACK" && body == new_hexId) {
              customPrint("WRITE ACK", rank_, backendId_);
              return;
          }
      }
  }

  customPrint("[CTL] CONFIG_ACK timeout", rank_, backendId_);
}

void BackendOpus::topowrite(int cnt) {
  // Send CONFIG_REQ
  if (is_emulation == 1) {
    // Emulate latency
    std::ostringstream oss;
    oss << "[CTL] Emulating provisioning latency of " << emulated_latency_ms << " ms";
    customPrint(oss.str(), rank_, backendId_);
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(emulated_latency_ms)));
    return;
  }

  int backend = backendId_;

  if (!is_pp_backend_) {
    cnt += pp_stage_ * PP_INTERVAL;
  } else {
    backend += PP_BACKEND_INTERVAL * (cnt / PP_INTERVAL);
  }
  
  std::string msg = "CONFIG_REQ;" + hexId_ + "," + std::to_string(backend) 
    + "," + std::to_string(cnt) + "\n";

  net::send(ctrlSock_, msg.data(), msg.size(), 0);
  // printf("[SENT TO CTL] %s\n", msg.c_str());
  customPrint("WRITE topo", rank_, backendId_);

  // Wait for CONFIG_ACK
  char buf[512];
  std::string accum;
  // printf("[DEBUG] Waiting for CONFIG_ACK on socket %d\n", ctrlSock_);
  
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(10000); // total wait

  while (std::chrono::steady_clock::now() < deadline) {

      ssize_t n = net::recv(ctrlSock_, buf, sizeof(buf), 0);

      if (n > 0) {
          accum.append(buf, n);
      }
      else if (n == 0) {
          customPrint("[CTL] Controller closed connection", rank_, backendId_);
          return;
      }
      else { // n < 0
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
              // timeout slice expired → retry
              continue;
          } else {
              perror("recv");
              customPrint("[CTL] recv failed", rank_, backendId_);
              return;
          }
      }

      // process all complete lines
      size_t pos;
      while ((pos = accum.find('\n')) != std::string::npos) {
          std::string line = accum.substr(0, pos);
          accum.erase(0, pos + 1);

          auto sep = line.find(';');
          std::string hdr  = sep == std::string::npos ? line : line.substr(0, sep);
          std::string body = sep == std::string::npos ? ""   : line.substr(sep + 1);

          if (hdr == "CONFIG_ACK" && body == hexId_) {
              customPrint("WRITE ACK", rank_, backendId_);
              return;
          }
      }
  }

  customPrint("[CTL] CONFIG_ACK timeout", rank_, backendId_);
}


void CUDART_CB post_coll_callback(void* opus_class) {
    // at::RecordFunction rf(at::RecordScope::USER_SCOPE);
    // const char* name = static_cast<const char*>(userData);
    // rf.before("col_done");

    BackendOpus* backendOpus = static_cast<BackendOpus*>(opus_class);
    backendOpus->post_coll_callback_helper();
}

void CUDART_CB pre_coll_callback(void* opus_class) {
    // at::RecordFunction rf(at::RecordScope::USER_SCOPE);
    // const char* name = static_cast<const char*>(userData);
    // rf.before("col_done");

    BackendOpus* backendOpus = static_cast<BackendOpus*>(opus_class);
    backendOpus->pre_coll_callback_helper();
}

void BackendOpus::groupEnd() {
  if (ncclActiveGroupCounter_ > 0) {
    ncclGroupEnd();
    --ncclActiveGroupCounter_;
  } else {
    throw std::runtime_error("Mismatched ncclGroupEnd call");
  }
}

void BackendOpus::groupStart() {
  ncclGroupStart();
  ++ncclActiveGroupCounter_;
}

void BackendOpus::startCoalescing() {
  groupStart();
}

 c10::intrusive_ptr<Work> BackendOpus::endCoalescing() {
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  
  groupEnd();
  // future->markCompleted(c10::IValue(std::vector<std::vector<at::Tensor>>{}));
  return c10::make_intrusive<WorkOpus>(OpType::COALESCED, std::move(future), stream_);
  
}
  

bool BackendOpus::is_in_comm_stage() {
  if (curCommStageIdx_ >= comm_stages_.size()) {
    return false;
  }
  if (is_pp_backend_) {
    // TODO implement this
    return false;
  }
  CommStage& stage = comm_stages_[curCommStageIdx_];
  if (backendId_ == stage.backendId_start &&
      collectiveCounter_ == stage.collCounter_start) {
    return false;
  }
  if (backendId_ == stage.backendId_end &&
      collectiveCounter_ == stage.collCounter_end) {
    curCommStageIdx_++;
    return false;
  }
  return true;
}

bool BackendOpus::is_comm_change_before(int collectiveCounter, int curCommStageIdx) {
  if (curCommStageIdx >= comm_stages_.size()) {
    return false;
  }
  CommStage& stage = comm_stages_[curCommStageIdx];
  int backendId_start = stage.backendId_start % PP_BACKEND_INTERVAL;
  if (backendId_ == backendId_start &&
      collectiveCounter == stage.collCounter_start) {
        return true;
  }
  return false;
}

bool BackendOpus::is_comm_change_after(int collectiveCounter, int curCommStageIdx) {
  if (curCommStageIdx >= comm_stages_.size()) {
    return false;
  }
  CommStage& stage = comm_stages_[curCommStageIdx];
  int backendId_end = stage.backendId_end % PP_BACKEND_INTERVAL;
  if (backendId_ == backendId_end &&
      collectiveCounter == stage.collCounter_end) {
        return true;
  }
  return false;
}

bool BackendOpus::check_comm_stage() {
  if (curCommStageIdx_ >= comm_stages_.size()) {
    return false;
  }
  CommStage& stage = comm_stages_[curCommStageIdx_];
  if (stage.backendId_start == 1 && 
    backendId_ != 1) {
    return false;
  } 
  if (stage.backendId_start == 2 || 
    stage.backendId_start == 5 ||
    stage.backendId_start == 6) {
      if (backendId_ != 2 &&
          backendId_ != 5 &&
          backendId_ != 6) {
            return false;
          }
  }
  return true;
}

void BackendOpus::pre_coll_callback_helper(int src_rank, int dst_rank) {
    // determine comm change before and after
    std::ostringstream counterStream;

    if (src_rank > rank_ || dst_rank > rank_) {
      counterStream << "PP comm group: " << get_pp_base(src_rank, dst_rank, rank_)
        << ", pre_coll, Idx: " << up_cnt_ + get_pp_base(src_rank, dst_rank, rank_) * PP_INTERVAL
        << ", start, curCommStageIdx_" << curCommStageIdx_;
    } else if (src_rank != -1 || dst_rank != -1) {
      counterStream << "PP comm group: " << get_pp_base(src_rank, dst_rank, rank_)
        << ", pre_coll, Idx: " << down_cnt_ + get_pp_base(src_rank, dst_rank, rank_) * PP_INTERVAL
        << ", start, curCommStageIdx_" << curCommStageIdx_;
    } else {
      counterStream << "pre_coll, Idx: " << collectiveCounter_ << ", start, curCommStageIdx_" << curCommStageIdx_;
    }
    customPrint(counterStream.str().c_str(), rank_, backendId_);

    {
      std::unique_lock lk(topo_mutex_);
      if (!topo_ready) {
          customPrint("pre_coll, topo not ready!", rank_, backendId_);
          topo_cv_.wait(lk, [this] { return topo_ready; });
      }
    }

    int cnt = 0;
    if (src_rank > rank_ || dst_rank > rank_) {
      cnt = up_cnt_ + get_pp_base(src_rank, dst_rank, rank_) * PP_INTERVAL;
    } else if (src_rank != -1 || dst_rank != -1) {
      cnt = down_cnt_ + get_pp_base(src_rank, dst_rank, rank_) * PP_INTERVAL;
    } else {
      cnt = collectiveCounter_;
    }

    bool comm_change_after = is_comm_change_after(cnt, curCommStageIdx_);
    bool comm_change_before = is_comm_change_before(cnt, curCommStageIdx_);
    
    if (domain_ == 0) {
        // scale-out domain
        if (mode == 1) {
            // provisioning
            // if (comm_change_before) {
                // bool valid_stage = check_comm_stage();
                // if (!valid_stage) {
                //     customPrint("Invalid comm stage detected!", rank_, backendId_);
                // }
                // std::unique_lock lk(topo_mutex_);
                // if (!topo_ready) {
                //     customPrint("pre_coll, comm changed, waiting for previous callback (provision)", rank_, backendId_);
                //     topo_cv_.wait(lk, [this] { return topo_ready; });
                // }
            // }
            if (comm_change_after) {
                std::unique_lock lk(topo_mutex_);
                customPrint("pre_coll, comm change detected, setting topo not ready", rank_, backendId_);
                topo_ready = false;
            } else {
                customPrint("pre_coll, no comm change after", rank_, backendId_);
            }
        } else if (mode == 0) {
            // default mode
            if (comm_change_before || is_pp_backend_) {
                // bool valid_stage = check_comm_stage();
                // if (!valid_stage) {
                //     customPrint("Invalid comm stage detected!", rank_, backendId_);
                // }
                // std::unique_lock lk(topo_mutex_);
                // if (!topo_ready) {
                //     customPrint("pre_coll, comm changed, waiting for previous callback, curCommStageIdx: " + std::to_string(curCommStageIdx), rank_, backendId_);
                //     topo_cv_.wait(lk, [this] { return topo_ready; });
                // }
                topowrite(cnt);
            } 
            if (comm_change_after) {
                std::unique_lock lk(topo_mutex_);
                customPrint("pre_coll, comm change detected, setting topo false", rank_, backendId_);
                topo_ready = false;
            } else {
                customPrint("pre_coll, no comm change detected", rank_, backendId_);
            }
        }
    }

    if (comm_change_after) {
      curCommStageIdx_++;
    }
    collectiveCounter_++;
    if (src_rank > rank_ || dst_rank > rank_) {
      up_cnt_++;
    } else if (src_rank != -1 || dst_rank != -1) {
      down_cnt_++;
    }
}

void BackendOpus::post_coll_callback_helper(int src_rank, int dst_rank) {
    std::ostringstream oss;

    if (src_rank > rank_ || dst_rank > rank_) {
      oss << "PP comm group: " << get_pp_base(src_rank, dst_rank, rank_)
        << ", post_coll, Idx: " << up_cnt_post_ + get_pp_base(src_rank, dst_rank, rank_) * PP_INTERVAL
        << ", end, curCommStageIdx_post_: " << curCommStageIdx_post_;
    } else if (src_rank != -1 || dst_rank != -1) {
      oss << "PP comm group: " << get_pp_base(src_rank, dst_rank, rank_)
        << ", post_coll, Idx: " << down_cnt_post_ + get_pp_base(src_rank, dst_rank, rank_) * PP_INTERVAL
        << ", end, curCommStageIdx_post_: " << curCommStageIdx_post_;
    } else {
      oss << "post_coll, Idx: " << collectiveCounter_post_ << ", end, curCommStageIdx_post_: " << curCommStageIdx_post_;
    }

    customPrint(oss.str(), rank_, backendId_);

    // if (mode == 2) {
    //     // bypass mode, no-ctl
    //     return;
    // }

    int cnt = 0;
      if (src_rank > rank_ || dst_rank > rank_) {
        cnt = up_cnt_post_ + get_pp_base(src_rank, dst_rank, rank_) * PP_INTERVAL;
      } else if (src_rank != -1 || dst_rank != -1) {
        cnt = down_cnt_post_ + get_pp_base(src_rank, dst_rank, rank_) * PP_INTERVAL;
      } else {
        cnt = collectiveCounter_post_;
    }
    bool comm_change_after = is_comm_change_after(cnt, curCommStageIdx_post_);

    if (comm_change_after) {
      customPrint("post_coll, comm change after, curCommStageIdx: " + std::to_string(curCommStageIdx_post_), rank_, backendId_);
      curCommStageIdx_post_++;

      if (mode == 1) {
          // provisioning mode
          topoprovision();
      }
      signalTopoReady(); 
    } else if (is_pp_backend_ && mode == 1) {
      topowrite(cnt + 1);
    } 

    collectiveCounter_post_++;
    if (src_rank > rank_ || dst_rank > rank_) {
      up_cnt_post_++;
    } else if (src_rank != -1 || dst_rank != -1) {
      down_cnt_post_++;
    }
}

void BackendOpus::pre_coll(std::string coll_type, int input_size, int output_size, int src_rank, int dst_rank) {
  // TODO determine coll group/backend change
  // If no change, do not register callback
	// If change, register callback, block next coll issue until callback finishes
  std::ostringstream counterStream;
  if (src_rank > rank_ || dst_rank > rank_) {
    counterStream << "PP comm group: " << get_pp_base(src_rank, dst_rank, rank_)  
      << ", Idx: " << up_cnt_main_ + get_pp_base(src_rank, dst_rank, rank_) * PP_INTERVAL
      << ", issuing " << coll_type << ", mode: " << mode << ", input_size: " << input_size
      << ", output_size: " << output_size;
  } else if (src_rank != -1 || dst_rank != -1) {
    counterStream << "PP comm group: " << get_pp_base(src_rank, dst_rank, rank_) 
      << ", Idx: " << down_cnt_main_ + get_pp_base(src_rank, dst_rank, rank_) * PP_INTERVAL
      << ", issuing " << coll_type << ", mode: " << mode << ", input_size: " << input_size
      << ", output_size: " << output_size;
  } else {
    counterStream << "Idx: " << collectiveCounter_main_ 
      << ", issuing " << coll_type << ", mode: " << mode << ", input_size: " << input_size
      << ", output_size: " << output_size;
  }
  

  customPrint(counterStream.str().c_str(), rank_, backendId_);
  
  if (is_pp_backend_) {
    if (src_rank > rank_ || dst_rank > rank_) {
      up_cnt_main_++;
    } else if (src_rank != -1 || dst_rank != -1) {
      down_cnt_main_++;
    }
  }

  collectiveCounter_main_++;

  if (network_type == 0 || (is_pp_backend_ && coll_type == "allreduce")) {
    // cpu frontend network
    return;
  }

  if (domain_ != 0) {
    // scale-up
    return;
  }

  // cudaLaunchHostFunc(stream_, pre_coll_callback, (void*)this);

   // Create CUDA event to know when to run pre-logic
  // cudaEvent_t event;
  // cudaEventCreate(&event);
  // cudaEventRecord(event, stream_);
  
  // // Queue pre-logic on separate thread
  // {
  //     std::lock_guard<std::mutex> lock(preMutex_);
  //     preQueue_.push([this, event]() {
  //         cudaEventSynchronize(event);  // Wait for GPU
  //         pre_coll_callback_helper();
  //         cudaEventDestroy(event);
  //     });
  // }
  // preCV_.notify_one();
  struct CallbackData {
    BackendOpus* opus_class;
    int src_rank;
    int dst_rank;
  };
  if (mode == 1) {
    // provisioning mode
    if (use_cuda_stream_ == 1 || use_cuda_stream_ == 3) {
      CallbackData* data = new CallbackData{this, src_rank, dst_rank};
      cudaLaunchHostFunc(stream_, [](void* userData) {
          CallbackData* data = static_cast<CallbackData*>(userData);
          data->opus_class->pre_coll_callback_helper(data->src_rank, data->dst_rank);
          delete data; // Clean up the allocated memory
      }, data);
    } else {
      pre_coll_callback_helper(src_rank, dst_rank);
    }
  } else {
    // default mode or no_ctl mode
      CallbackData* data = new CallbackData{this, src_rank, dst_rank};
      cudaLaunchHostFunc(stream_, [](void* userData) {
          CallbackData* data = static_cast<CallbackData*>(userData);
          data->opus_class->pre_coll_callback_helper(data->src_rank, data->dst_rank);
          delete data; // Clean up the allocated memory
      }, data);
  }
}

void BackendOpus::post_coll(std::string coll_type, int src_rank, int dst_rank) {
  if (network_type == 0 || (is_pp_backend_ && coll_type == "allreduce")) {
    // cpu frontend network
    return;
  }
  
  if (domain_ != 0) {
    // scale-up
    return;
  }
  
  // Helper lambda for callback logic - avoids code duplication
  auto enqueue_callback = [this](int src, int dst) {
    struct CallbackData {
      BackendOpus* opus_class;
      int src_rank;
      int dst_rank;
    };
    
    CallbackData* data = new CallbackData{this, src, dst};
    
    cudaError_t err = cudaLaunchHostFunc(stream_, [](void* userData) {
      CallbackData* data = static_cast<CallbackData*>(userData);
      data->opus_class->post_coll_callback_helper(data->src_rank, data->dst_rank);
      delete data;
    }, data);
    
    // Critical: Clean up if launch failed
    if (err != cudaSuccess) {
      delete data;
      // Handle error appropriately
      throw std::runtime_error("cudaLaunchHostFunc failed: " + 
                               std::string(cudaGetErrorString(err)));
    }
  };
  
  if (mode == 1) {
    // provisioning mode
    if (use_cuda_stream_ == 2 || use_cuda_stream_ == 3) {
      enqueue_callback(src_rank, dst_rank);
    } else {
      cudaEvent_t event;
      cudaError_t err = cudaEventCreate(&event);
      if (err != cudaSuccess) {
        throw std::runtime_error("cudaEventCreate failed");
      }
      
      err = cudaEventRecord(event, stream_);
      if (err != cudaSuccess) {
        cudaEventDestroy(event);
        throw std::runtime_error("cudaEventRecord failed");
      }
      
      // Queue post-logic on separate thread
      {
        std::lock_guard<std::mutex> lock(postMutex_);
        postQueue_.push([this, event, src_rank, dst_rank]() {
          cudaEventSynchronize(event);
          post_coll_callback_helper(src_rank, dst_rank);
          cudaEventDestroy(event);
        });
      }
      postCV_.notify_one();
    }
  } else {
    enqueue_callback(src_rank, dst_rank);
  }
}

void BackendOpus::populate_comm_stages() {
  const char* file_name_env = std::getenv("COMM_PATTERN_PATH");
  if (!file_name_env) {
    throw std::runtime_error("Environment variable COMM_PATTERN_PATH not set");
  }

  std::ostringstream file_name_stream;
  file_name_stream << file_name_env << "_" << nodeRank_ << "_rank" << localRank_ << ".txt";
  std::string file_name = file_name_stream.str();
  std::ifstream infile(file_name);

  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open" + file_name);
  }

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#' || line[0] == 'N') continue;

    std::istringstream iss(line);

    std::string type, tmp;
    int start_backend, start_idx, end_backend, end_idx, len;

    // remove trailing comma from type
    iss >> type;

    if (!(iss
          >> tmp >> start_backend        // start_backend
          >> tmp >> start_idx            // start_idx
          >> tmp >> end_backend          // end_backend
          >> tmp >> end_idx              // end_idx
          >> tmp >> len)) {              // len
        throw std::runtime_error("Invalid format in comm_stages.txt");
    }

    comm_stages_.push_back({start_backend, start_idx, end_backend, end_idx});

    // std::cout << "Parsed stage: " << type
    //           << " start(" << start_backend << ", " << start_idx << ") "
    //           << "end(" << end_backend << ", " << end_idx << ") "
    //           << "len: " << len << std::endl;
  }
  infile.close();
}

void BackendOpus::preThreadFunc() {
    while (running_) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(preMutex_);
            preCV_.wait(lock, [this] { return !preQueue_.empty() || !running_; });
            if (!running_ && preQueue_.empty()) break;
            if (!preQueue_.empty()) {
                task = std::move(preQueue_.front());
                preQueue_.pop();
            }
        }
        if (task) task();
    }
}

void BackendOpus::postThreadFunc() {
    while (running_) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(postMutex_);
            postCV_.wait(lock, [this] { return !postQueue_.empty() || !running_; });
            if (!running_ && postQueue_.empty()) break;
            if (!postQueue_.empty()) {
                task = std::move(postQueue_.front());
                postQueue_.pop();
            }
        }
        if (task) task();
    }
}

BackendOpus::BackendOpus(int rank, int size, c10::intrusive_ptr<Options> options)
    : Backend(rank, size) {
  
  // TODO custom backend handling
  backendId_ = backend_count;

  // backendId_ = 1;
  backend_count++;

  // customPrint(options->group_name, rank, backendId_);
  std::ostringstream oss;
  // oss << "Global ranks in group: ";
  // for (const auto& rank : options->global_ranks_in_group) {
  //   oss << rank << " ";
  // }
  // customPrint(oss.str(), rank, backendId_);

  size_ = size;
  rank_ = rank;

  int global_rank = std::stoi(std::getenv("RANK"));
  customPrint("Global rank: " + std::to_string(global_rank) + ", pg_uid_: " + pg_uid_, rank_, backendId_);

  localRank_ = rank;

  const char* local_rank_env = std::getenv("LOCAL_RANK");
  if (local_rank_env != nullptr) {
    localRank_ = std::stoi(local_rank_env);
    oss.str("");
    oss << "LOCAL_RANK: " << localRank_ << "\n";
    customPrint(oss.str().c_str(), rank_, backendId_);
    cudaError_t err = cudaSetDevice(localRank_);
    if (err != cudaSuccess) {
      printf("Failed to set CUDA device: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  } else {
    printf("Environment variable LOCAL_RANK not set. Using default device ID.\n");
    cudaError_t err = cudaSetDevice(rank);
    if (err != cudaSuccess) {
      printf("Failed to set CUDA device: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  
  // TODO we allocate all the communication to default stream

  // TODO dynamically select server IP using controller
  const char* server_ips_env = std::getenv("SERVER_IPS");
  const char* hostname_env = std::getenv("HOSTNAME");
  if (!hostname_env) {
    throw std::runtime_error("Environment variable HOSTNAME not set");
  }

  std::string hostname(hostname_env);
  std::istringstream server_ips_stream(server_ips_env);
  std::string ip;
  nodeRank_ = -1;
  int index = 0;

  while (std::getline(server_ips_stream, ip, ',')) {
    if (hostname == ip) {
      nodeRank_ = index;
      break;
    }
    ++index;
  }

  if (nodeRank_ == -1) {
    throw std::runtime_error("Hostname not found in SERVER_IPS");
  }
  customPrint("Node rank: " + std::to_string(nodeRank_), rank_, backendId_);

  if (server_ips_env != nullptr) {
    SERVER_IPS.clear();
    std::istringstream iss(server_ips_env);
    std::string ip;
    while (std::getline(iss, ip, ',')) {
      SERVER_IPS.push_back(ip);
    }
  }

  std::string server_ip = SERVER_IPS[0];
  int port = 34567;
  
  // general impl for 2D parallel
  determine_server_addr(server_ip, port, global_rank, localRank_, backendId_, SERVER_IPS);

  oss.str("");
  oss << "MASTER_ADDR: " << server_ip << ", port: " << port << ", size: " << size_ << "\n";
  customPrint(oss.str().c_str(), rank_, backendId_);

  int clientSockets[size_-1];
  int sockfd;

  if (rank_ == 0) {
    // Rank 0 acts as the server
    sockfd = createServerSocket(server_ip.c_str(), port);

    printf("Rank 0 waiting for %d connections...\n", size_ - 1);

    for (int i = 0; i < size_ - 1; i++) {
      clientSockets[i] = accept(sockfd, NULL, NULL);
      if (clientSockets[i] < 0) {
          perror("Error accepting connection");
          exit(EXIT_FAILURE);
      }
      printf("Accepted connection %d\n", i + 1);
    }
  } else {
    int attempts = 0;
    const int max_attempts = 5;

    while (attempts < max_attempts) {
      sockfd = createClientSocket(server_ip.c_str(), port);  // Attempt to connect to the server
      if (sockfd >= 0) {
        printf("Connected to %s:%d\n", server_ip.c_str(), port);
        break;
      }
      attempts++;
      printf("Connection attempt %d failed. Retrying...\n", attempts);
      sleep(1);  // Wait for 1 second before retrying
    }

    if (sockfd < 0) {
      throw std::runtime_error("Failed to connect to server after 5 attempts");
    }
  }

  // set NCCL environment variable
  // TODO. 1D parallelism only has one backendID
  if (backendId_ == 0) {
    if (is_1d_parallelism()) {
      network_type = 1;
    } else {
      network_type = 0;
    }
  } else {
    network_type = 1;
  }


  if (rank_ == 0) {
    ncclGetUniqueId(&id_);

    for (int i = 0; i < size_ - 1; i++) {
      net::send(clientSockets[i], &id_, sizeof(id_), 0);
      close(clientSockets[i]);
    }

    close(sockfd);
  } else {
    net::recv(sockfd, &id_, sizeof(id_), 0);
    close(sockfd);
  }

  // if (network_type == 1) {
    // gpu backend network
  checkin(); // checkin regardless of network type
  // }
  
  // assume id has been broadcast to all ranks

  NCCLCHECK(ncclCommInitRank(&comm_, size_, id_, rank_));
  CUDACHECK(cudaStreamCreate(&stream_));

  collectiveCounter_ = 0;
  collectiveCounter_post_ = 0;
  collectiveCounter_main_ = 0;
  up_cnt_ = 0;
  down_cnt_ = 0;
  up_cnt_post_ = 0;
  down_cnt_post_ = 0;
  up_cnt_main_ = 0;
  down_cnt_main_ = 0;
  

  const char* mode_env = std::getenv("MODE");
  if (mode_env != nullptr) {
    std::string mode_str(mode_env);
    if (mode_str == "baseline") {
      mode = 0;
    } else if (mode_str == "provision") {
      mode = 1;
    } else if (mode_str == "no-ctl") {
      mode = 2;
    } else {
      throw std::runtime_error("Invalid MODE environment variable value. Expected 'baseline' or 'provision'.");
    }
  }

  is_pp_backend_ = is_pp_enabled() && (backendId_ == 1);
  if (is_pp_backend_) {
    pp_stage_ = rank_; // thread safe, as each backend is launched in the same thread
  }

  const char* use_cuda_stream_env = std::getenv("USE_CUDA_STREAM");
  if (use_cuda_stream_env != nullptr) {
    std::string use_cuda_stream_str(use_cuda_stream_env);
    use_cuda_stream_ = std::stoi(use_cuda_stream_str);
    if (use_cuda_stream_ < 0 || use_cuda_stream_ > 3) {
      throw std::runtime_error("Invalid USE_CUDA_STREAM environment variable value. Expected '0', '1', '2', or '3'");
    }
  }

  if (mode != 2) {
    populate_comm_stages();
  }

  const char* emu_env = std::getenv("IS_EMULATION");
  if (emu_env != nullptr) {
    std::string emu_str(emu_env);
    if (emu_str == "1") {
      is_emulation = 1;
      const char* emu_time = std::getenv("RECONFIG_LATENCY");
      if (emu_time != nullptr) {
        emulated_latency_ms = std::stof(emu_time);
        customPrint("Emulation time set to " + std::to_string(emulated_latency_ms) + " ms", rank_, backendId_);
      }
    } else {
      is_emulation = 0;
    }
  }

  preThread_ = std::thread(&BackendOpus::preThreadFunc, this);
  postThread_ = std::thread(&BackendOpus::postThreadFunc, this);
}

BackendOpus::~BackendOpus() {
  ncclCommDestroy(comm_);
  CUDACHECK(cudaStreamDestroy(stream_));
  // TODO release resource in the controller if network_type == 1
  running_ = false;
  preCV_.notify_all();
  postCV_.notify_all();
  if (preThread_.joinable()) preThread_.join();
  if (postThread_.joinable()) postThread_.join();
}

// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendOpus::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions&) {
  
  pre_coll("allgather", inputTensors[0].numel(), outputTensors[0].size() * outputTensors[0][0].numel());
  
  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    at::Tensor& inputTensor = inputTensors[0];
    at::Tensor outputTensor = at::empty({size_, inputTensor.numel()}, inputTensor.options());

    // Perform NCCL AllGather operation
    NCCLCHECK(ncclAllGather(
      inputTensor.data_ptr(),
      outputTensor.data_ptr(),
      inputTensor.numel(),
      opus_utils::getNcclDataType(inputTensor.scalar_type()),
      comm_,
      stream_));

    for (int i = 0; i < size_; ++i) {
        // Copy the i-th row from the 2D output tensor
        outputTensors[0][i].copy_(outputTensor[i]);
    }
  }

  post_coll("allgather");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  
  future->markCompleted(c10::IValue(outputTensors));
  auto work = c10::make_intrusive<WorkOpus>(OpType::ALLGATHER, std::move(future), stream_);
  return work;
}

c10::intrusive_ptr<Work> BackendOpus::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const AllgatherOptions& /* unused */) {
  
  pre_coll("_allgather_base", inputTensor.numel(), outputTensor.numel());

  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    // Check the size of inputTensor and outputTensor
    if (inputTensor.numel() * size_ != outputTensor.numel()) {
      throw std::runtime_error("Output tensor size must be equal to input tensor size multiplied by the number of ranks");
    }

    // Perform NCCL AllGather operation
    NCCLCHECK(ncclAllGather(
      inputTensor.data_ptr(),
      outputTensor.data_ptr(),
      inputTensor.numel(),
      opus_utils::getNcclDataType(inputTensor.scalar_type()),
      comm_,
      stream_));

    post_coll("_allgather_base");
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensor));
  return c10::make_intrusive<WorkOpus>(OpType::ALLGATHER, std::move(future), stream_);
}


// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendOpus::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions&) {
  
  pre_coll("allgather_coalesced", 
    inputTensors[0].numel() * inputTensors.size(), 
    outputTensors[0].numel() * outputTensors.size());

  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    for (size_t i = 0; i < outputTensors.size(); ++i) {
      at::Tensor& inputTensor = inputTensors[i];
      at::Tensor& outputTensor = outputTensors[i];

      // Perform NCCL AllGather operation
      NCCLCHECK(ncclAllGather(
        inputTensor.data_ptr(),
        outputTensor.data_ptr(),
        inputTensor.numel(),
        opus_utils::getNcclDataType(inputTensor.scalar_type()),
        comm_,
        stream_));
    }
  }
  
  post_coll("allgather_coalesced");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkOpus>(OpType::ALLGATHER, std::move(future), stream_);
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendOpus::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  // TODO support two dim tensors
  
  // at::RecordFunction rf(at::RecordScope::USER_SCOPE);
  // rf.before("topowrite");
  // {
  //   // nvtx3::scoped_range topowrite_range("topowrite");
  //   topowrite();   // <-- this call is now profiled
  // } 
  // rf.end();
  pre_coll("allreduce", tensors[0].numel(), tensors[0].numel());

  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    // nvtx3::scoped_range topowrite_range("nccl_allreduce");
    NCCLCHECK(
      ncclAllReduce(tensors[0].data_ptr(), 
      tensors[0].data_ptr(), 
      tensors[0].numel(), 
      opus_utils::getNcclDataType(tensors[0].scalar_type()),
      ncclSum, 
      comm_, 
      stream_));
  }

  post_coll("allreduce");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkOpus>(OpType::ALLREDUCE, std::move(future), stream_);
}

c10::intrusive_ptr<Work> BackendOpus::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendOpus::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opt) {

  // if (inputTensors.size() != size_) {
  //   throw std::runtime_error("sendbuff must contain exactly world_size tensors for scatter, got " + std::to_string(inputTensors[0].size()));
  // }

  pre_coll("alltoall",
           /*in=*/inputTensors[0].numel() * inputTensors.size(),
           /*out=*/outputTensors[0].numel() * outputTensors.size());

  auto dtype = opus_utils::getNcclDataType(inputTensors[0].scalar_type());

  ncclGroupStart();
  for (int r = 0; r < size_; r++) {
    auto& send = inputTensors[r];
    auto& recv = outputTensors[r];

    // TORCH_CHECK(send.numel() == recv.numel(),
    //             "Send/recv tensor size mismatch in alltoall");
    ncclSend(send.data_ptr(), send.numel(), dtype, r, comm_, stream_);
    ncclRecv(recv.data_ptr(), recv.numel(), dtype, r, comm_, stream_);
    
    
  }
  ncclGroupEnd();

  post_coll("alltoall");
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(outputTensors));

  return c10::make_intrusive<WorkOpus>(OpType::ALLTOALL, std::move(future), stream_);
}

inline void* ptr_with_offset(at::Tensor& t, int64_t offset_elems) {
  return static_cast<char*>(t.data_ptr()) +
         offset_elems * t.element_size();
}

c10::intrusive_ptr<Work> BackendOpus::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {

  // if (inputSplitSizes.size() != size_ ||
  //     outputSplitSizes.size() != size_) {
  //   throw std::runtime_error(
  //       "Input and output split sizes must have a length equal to world size");
  // }

  std::vector<int64_t> inputSplitSizes_(size_, inputTensor.numel() / size_);
  std::vector<int64_t> outputSplitSizes_(size_, outputTensor.numel() / size_);

  pre_coll("alltoall_base",
          inputTensor.numel(),
          outputTensor.numel());

  auto dtype = opus_utils::getNcclDataType(inputTensor.scalar_type());

  int64_t sendOffset = 0;
  int64_t recvOffset = 0;

  ncclGroupStart();
  for (int r = 0; r < size_; r++) {
    int64_t sendCount = inputSplitSizes_[r];
    int64_t recvCount = outputSplitSizes_[r];

    ncclSend(ptr_with_offset(inputTensor, sendOffset), 
    sendCount, dtype, r, comm_, stream_);

    ncclRecv(ptr_with_offset(outputTensor, recvOffset),
      recvCount, dtype, r, comm_, stream_);
    sendOffset += sendCount;
    recvOffset += recvCount;
  }
  ncclGroupEnd();

  post_coll("alltoall_base");
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(outputTensor));

  return c10::make_intrusive<WorkOpus>(OpType::ALLTOALL_BASE, std::move(future), stream_);
}

c10::intrusive_ptr<Work> BackendOpus::barrier(
  const BarrierOptions& /* unused */) {

  pre_coll("barrier", 1, 1);

  at::Tensor dummyTensor = at::zeros({1}, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    // Use NCCL AllReduce with a dummy tensor to synchronize all ranks
    NCCLCHECK(ncclAllReduce(
      dummyTensor.data_ptr(),
      dummyTensor.data_ptr(),
      dummyTensor.numel(),
      opus_utils::getNcclDataType(dummyTensor.scalar_type()),
      ncclSum,
      comm_,
      stream_));
  }

  post_coll("barrier");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  if (network_type == 0) {
    future->markCompleted(c10::IValue(std::vector<at::Tensor>{}));
  } else {
    future->markCompleted(c10::IValue(std::vector<at::Tensor>{dummyTensor}));
  }
  return c10::make_intrusive<WorkOpus>(OpType::BARRIER, std::move(future), stream_);
}

c10::intrusive_ptr<Work> BackendOpus::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {

  pre_coll("broadcast", tensors[0].numel(), tensors[0].numel());

  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    at::Tensor& buffer = tensors[0];

    NCCLCHECK(ncclBcast(
        buffer.data_ptr(),
        buffer.numel(),
        opus_utils::getNcclDataType(buffer.scalar_type()),
        opts.rootRank,
        comm_,
        stream_));
  }

  post_coll("broadcast");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkOpus>(OpType::BROADCAST, std::move(future), stream_);
}

c10::intrusive_ptr<Work> BackendOpus::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendOpus::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendOpus::reduce_scatter(
    std::vector<at::Tensor>& recvbuff,
    std::vector<std::vector<at::Tensor>>& sendbuff,
    const ReduceScatterOptions& opts) {
  // TODO:: verify this implementation of reducescatter
  pre_coll("reduce_scatter", 
    sendbuff[0].size() * sendbuff[0][0].numel(), 
    recvbuff[0].numel());

  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    if (sendbuff[0].size() != size_) {
      throw std::runtime_error("sendbuff must contain exactly world_size tensors for reduce_scatter, got " + std::to_string(sendbuff[0].size()));
    }

    at::Tensor& outputTensor = recvbuff[0];
    at::Tensor inputTensor = at::cat(sendbuff[0], 0);


    if (inputTensor.numel() != outputTensor.numel() * size_) {
      throw std::runtime_error("Input tensor size must be equal to output tensor size multiplied by the number of ranks");
    }

    NCCLCHECK(ncclReduceScatter(
        inputTensor.data_ptr(),
        outputTensor.data_ptr(),
        outputTensor.numel(),
        opus_utils::getNcclDataType(inputTensor.scalar_type()),
        ncclSum,
        comm_,
        stream_));

  }


  post_coll("reduce_scatter");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(recvbuff));
  return c10::make_intrusive<WorkOpus>(OpType::REDUCE_SCATTER, std::move(future), stream_);
  
}

c10::intrusive_ptr<Work> BackendOpus::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& /* unused */) {

  pre_coll("_reduce_scatter_base", 
    inputTensor.numel(), 
    outputTensor.numel());

  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    if (inputTensor.numel() != outputTensor.numel() * size_) {
      throw std::runtime_error("Input tensor size must be equal to output tensor size multiplied by the number of ranks");
    }

    NCCLCHECK(ncclReduceScatter(
      inputTensor.data_ptr(),
      outputTensor.data_ptr(),
      outputTensor.numel(),
      opus_utils::getNcclDataType(inputTensor.scalar_type()),
      ncclSum,
      comm_,
      stream_));
  }


  post_coll("_reduce_scatter_base");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(outputTensor));
  return c10::make_intrusive<WorkOpus>(OpType::REDUCE_SCATTER, std::move(future), stream_);
}

c10::intrusive_ptr<Work> BackendOpus::reduce_scatter_tensor_coalesced(
  std::vector<at::Tensor>& outputTensors,
  std::vector<at::Tensor>& inputTensors,
  const ReduceScatterOptions& opts) {
  // TODO:: verify this implementation of reducescatter
  // TODO: perform coalescing

  pre_coll("reduce_scatter_coalesced", 
    inputTensors[0].numel() * inputTensors.size(), 
    outputTensors[0].numel() * outputTensors.size());

  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    for (size_t i = 0; i < inputTensors.size(); ++i) {

      at::Tensor& outputTensor = outputTensors[i];
      at::Tensor inputTensor = inputTensors[i];

      if (inputTensor.numel() != outputTensor.numel() * size_) {
        throw std::runtime_error("Input tensor size must be equal to output tensor size multiplied by the number of ranks");
      }

      NCCLCHECK(ncclReduceScatter(
        inputTensor.data_ptr(),
        outputTensor.data_ptr(),
        outputTensor.numel(),
        opus_utils::getNcclDataType(inputTensor.scalar_type()),
        ncclSum,
        comm_,
        stream_));
    }
  }

  post_coll("reduce_scatter_coalesced");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkOpus>(OpType::REDUCE_SCATTER, std::move(future), stream_);
}


c10::intrusive_ptr<Work> BackendOpus::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opt) {
  
  pre_coll("scatter", 
    inputTensors[0].size() * inputTensors[0][0].numel(), 
    outputTensors[0].numel());

  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    if (inputTensors[0].size() != size_) {
      throw std::runtime_error("sendbuff must contain exactly world_size tensors for scatter, got " + std::to_string(inputTensors[0].size()));
    }
    
    at::Tensor& outputTensor = outputTensors[0];
    at::Tensor inputTensor = at::cat(inputTensors[0], 0);


    if (inputTensor.numel() != outputTensor.numel() * size_) {
      throw std::runtime_error("Input tensor size must be equal to output tensor size multiplied by the number of ranks");
    }

    // TODO: implement scatter

    // Wrap input/output in c10d tensors
    // std::vector<at::Tensor> output_vec = {outputTensor};
    // std::vector<at::Tensor> input_vec = {inputTensor};

    // // Create scatter work options
    // c10d::ScatterOptions gloo_opt;
    // gloo_opt.rootRank = opt.rootRank;
    // gloo_opt.rootTensor = inputTensor;
    // gloo_opt.tensor = outputTensor;

    // // Call Gloo scatter
    // pg_->scatter(outputTensor, inputTensor, gloo_opt);

  }


  post_coll("scatter");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkOpus>(OpType::SCATTER, std::move(future), stream_);

}

c10::intrusive_ptr<Work> BackendOpus::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {

  pre_coll("send", tensors[0].numel(), 0, -1, dstRank);

  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    ncclGroupStart();
    NCCLCHECK(ncclSend(
        tensors[0].data_ptr(),
        tensors[0].numel(),
        opus_utils::getNcclDataType(tensors[0].scalar_type()),
        dstRank,
        comm_,
        stream_));
    ncclGroupEnd();
  }

  post_coll("send", -1, dstRank);
  
  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  // future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkOpus>(OpType::SEND, std::move(future), stream_);

}

c10::intrusive_ptr<Work> BackendOpus::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {

  pre_coll("recv", 0, tensors[0].numel(), srcRank, -1);

  if (network_type == 0) {
    // TODO cpu frontend network
  } else {
    ncclGroupStart();

    NCCLCHECK(ncclRecv(
        tensors[0].data_ptr(),
        tensors[0].numel(),
        opus_utils::getNcclDataType(tensors[0].scalar_type()),
        srcRank,
        comm_,
        stream_));
    ncclGroupEnd();

  }

  post_coll("recv", srcRank, -1);

  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  // future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkOpus>(OpType::RECV, std::move(future), stream_);

}

c10::intrusive_ptr<Work> BackendOpus::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Backend> BackendOpus::createBackendOpus(
    const c10::intrusive_ptr<::c10d::Store>& /* unused */,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<BackendOpus>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createBackendOpus", &BackendOpus::createBackendOpus);
}

} // namespace c10d
