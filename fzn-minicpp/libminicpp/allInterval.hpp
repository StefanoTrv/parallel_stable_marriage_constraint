#ifndef __ALLINTERVAL_H
#define __ALLINTERVAL_H

#include <set>
#include <array>
#include <map>
#include <algorithm>
#include <iomanip>
#include <stdint.h>
#include "intvar.hpp"
#include "mddstate.hpp"

// Dedicated constraints to compare domain propagation to MDD propagation

class EQAbsDiffBC : public Constraint { // z == |x - y|
  var<int>::Ptr _z,_x,_y;
public:
  EQAbsDiffBC(var<int>::Ptr z,var<int>::Ptr x,var<int>::Ptr y)
       : Constraint(x->getSolver()),_z(z),_x(x),_y(y) {}
   void post() override;
};

namespace Factory {
  inline Constraint::Ptr equalAbsDiff(var<int>::Ptr z,var<int>::Ptr x,var<int>::Ptr y) {
    return new (x->getSolver()) EQAbsDiffBC(z,x,y);
  }

  void absDiffMDD(MDDSpec& mdd, const Factory::Veci& vars);
}





#endif
