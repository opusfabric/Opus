/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ALL_TO_ALLV_LOAD_HH__
#define __ALL_TO_ALLV_LOAD_HH__

#include <cstdint>
#include <map>
#include <memory>
#include <utility>

namespace AstraSim {

using AllToAllVLoadMap = std::map<std::pair<int, int>, uint64_t>;
using AllToAllVLoadMapPtr = std::shared_ptr<const AllToAllVLoadMap>;

}  // namespace AstraSim

#endif /* __ALL_TO_ALLV_LOAD_HH__ */
