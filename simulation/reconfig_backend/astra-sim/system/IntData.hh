/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __INT_DATA_HH__
#define __INT_DATA_HH__

namespace AstraSim {

class IntData : public CallData {
  public:
    IntData(int d) {
        data = d;
    }
    int data;
    uint64_t execution_time;
};


class TwoIntData : public CallData {
  public:
    TwoIntData(int d1, int d2) {
        data = d1;
        data2 = d2;
    }
    int data;
    int data2;
    uint64_t execution_time;
};

}  // namespace AstraSim

#endif /* __INT_DATA_HH__ */
