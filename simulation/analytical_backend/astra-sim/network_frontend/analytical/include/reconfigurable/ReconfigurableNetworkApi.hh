/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CommonNetworkApi.hh"
#include <astra-network-analytical/reconfigurable/Topology.h>
#include <astra-network-analytical/reconfigurable/TopologyManager.h>

#include <unordered_set>

using namespace AstraSim;
using namespace AstraSimAnalytical;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalReconfigurable;

namespace AstraSimAnalyticalReconfigurable {

class ReconfigurableNetworkApi final : public CommonNetworkApi {
  public:
    /**
     * Set the topology to be used.
     *
     * @param tm_ptr pointer to the topology
     */
    static void set_topology(std::shared_ptr<TopologyManager> tm_ptr) noexcept;

    /**
     * Constructor.
     *
     * @param rank id of the API
     */
    explicit ReconfigurableNetworkApi(int rank) noexcept;

    /**
     * Implement sim_send of AstraNetworkAPI.
     */
    int sim_send(void* buffer,
                 uint64_t count,
                 int type,
                 int dst,
                 int tag,
                 sim_request* request,
                 void (*msg_handler)(void* fun_arg),
                 void* fun_arg) override;

  bool sim_reconfig(int topo_id, int skip_inflight = 0) override;

  void increment_inflight_coll(int rank, std::string name) override;

  int get_inflight_coll() override;

  void decrement_inflight_coll(int rank, int node_id) override;

  void print_on_going_comms() override;

  private:
    /// topology
    static std::shared_ptr<TopologyManager> tm;

    static std::map<int, std::unordered_set<std::string>> on_going_comms;
};

}  // namespace AstraSimAnalyticalReconfigurable
