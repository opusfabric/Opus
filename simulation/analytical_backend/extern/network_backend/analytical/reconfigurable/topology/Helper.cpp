/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "reconfigurable/Helper.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalReconfigurable;

using namespace std;

std::shared_ptr<Topology> NetworkAnalyticalReconfigurable::construct_topology(
    const NetworkParser& network_parser) noexcept {
    // get network_parser info
    const auto dims_count = network_parser.get_dims_count();
    const auto topologies_per_dim = network_parser.get_topologies_per_dim();
    const auto npus_counts_per_dim = network_parser.get_npus_counts_per_dim();
    const auto latencies_per_dim = network_parser.get_latencies_per_dim();
    const auto reconfig_time = network_parser.get_reconfig_time();

    // for now, reconfigurable backend supports 1-dim topology only
    if (dims_count != 1) {
        std::cerr << "[Error] (network/analytical/reconfigurable) " << "only support 1-dim topology" << std::endl;
        std::exit(-1);
    }

    const auto npus_count = npus_counts_per_dim[0];

    return std::make_shared<Topology>(npus_count, npus_count);
}
