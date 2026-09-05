/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ALL_TO_ALLV_HH__
#define __ALL_TO_ALLV_HH__

#include "astra-sim/system/AllToAllVLoad.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/AllToAll.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/RingTopology.hh"

namespace AstraSim {

class AllToAllv : public AllToAll {
  public:
    AllToAllv(ComType type,
              int window,
              int id,
              RingTopology* allToAllTopology,
              uint64_t data_size,
              RingTopology::Direction direction,
              InjectionPolicy injection_policy,
              AllToAllVLoadMapPtr all2allv_loads,
              uint64_t all2allv_total_size,
              uint64_t all2allv_chunk_size);
    bool ready() override;
    uint64_t pair_msg_size(int src, int dst) const;

  private:
    AllToAllVLoadMapPtr all2allv_loads;
    uint64_t all2allv_total_size;
    uint64_t all2allv_chunk_size;
};

}  // namespace AstraSim

#endif /* __ALL_TO_ALLV_HH__ */
