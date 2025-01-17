/*
 * mini-cp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * mini-cp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2018. by Laurent Michel, Pierre Schaus, Pascal Van Hentenryck
 */

#ifndef __LEX_HPP
#define __LEX_HPP

#include <set>
#include <array>
#include <map>
#include <algorithm>
#include <iomanip>
#include <stdint.h>
#include "matrix.hpp"
#include "intvar.hpp"
#include "acstr.hpp"
#include "bitset.hpp"
#include "range.hpp"

class LexLeq : public Constraint {
   Factory::Veci _x;
   Factory::Veci _y;
   trail<int>    _q,_r,_s,_u;
   const int _sz;
   void listenFrom(int ofs);
   void propagateFrom(int k);
public:
   template <class Vec> LexLeq(const Vec& x,const Vec& y)
      : Constraint(x[0]->getSolver()),
        _x(x.size(),Factory::alloci(x[0]->getStore())),
        _y(y.size(),Factory::alloci(y[0]->getStore())),
        _q(x[0]->getSolver()->getStateManager(),0),
        _r(x[0]->getSolver()->getStateManager(),0),
        _s(x[0]->getSolver()->getStateManager(),0),
        _u(x[0]->getSolver()->getStateManager(),0),
        _sz((int)x.size())
   {
      int i = 0;
      for(auto& xi : x) _x[i++] = xi;
      i = 0;
      for(auto& yi : y) _y[i++] = yi;
   }
   void post() override;
};

namespace Factory {
   template <class Vec> Constraint::Ptr lexLeq(const Vec& x,const Vec& y) {
      if (x.size() != y.size()) {
         std::cout << "Lex requires arrays of identical sizes\n";
         abort();
         return nullptr;
      }
      return new (x[0]->getSolver()) LexLeq(x,y);
   }
}

#endif
