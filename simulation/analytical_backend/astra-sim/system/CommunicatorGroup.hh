/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMUNICATOR_GROUP_HH__
#define __COMMUNICATOR_GROUP_HH__

#include <assert.h>
#include <map>
#include <vector>

#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;
class CollectivePlan;
class CommunicatorGroup {
  public:
    CommunicatorGroup(int id, std::vector<int> involved_NPUs, Sys* generator);
    CollectivePlan* get_collective_plan(ComType comm_type);
    void set_id(int id);
    ~CommunicatorGroup();

    std::vector<int> involved_NPUs;
    int num_streams;
    std::string to_string() const {
        std::string result = "CG(id=" + std::to_string(id) + ", NPUs=[";
        for (size_t i = 0; i < involved_NPUs.size(); i++) {
            result += std::to_string(involved_NPUs[i]);
            if (i != involved_NPUs.size() - 1) {
                result += ", ";
            }
        }
        result += "])";
        return result;
    };

    int get_id() const {
        return id;
    };

    std::vector<int> compatible_topo;

  private:
    int id;
    Sys* generator;
    std::map<ComType, CollectivePlan*> comm_plans;
};

}  // namespace AstraSim

#endif /* __COMMUNICATOR_GROUP_HH__ */
