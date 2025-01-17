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

#include "regular.hpp"
#include "constraint.hpp"
#include "table.hpp"

std::vector<std::vector<int>> Automaton::transition() const
{
   std::vector<std::vector<int>> table(_t.size());
   int row = 0;
   for(const auto& t : _t) {
      table[row++] = std::vector<int> {t.from, t.symbol, t.to};
   }
   return table;
}

void Regular::post()
{
   auto cp = _A->getSolver();
   auto states = _A->getStates();
   auto finals = _A->getFinals();
   int start  = _A->getStart();
   int lb = 0,ub = _x.size();
   Factory::Veci q = Factory::intVarArray(cp,ub - lb + 1, states.first(),states.to());
   auto tr = _A->transition();
   var<int>::Ptr first = q[0];
   using namespace Factory;
   cp->post(first == start);
   for(int k=0;k < ub;k++)
      cp->post(table(std::vector<var<int>::Ptr> {q[k],_x[k],q[k+1]},tr));
   for(auto s : states) 
      if (finals.find(s) == finals.end())
         cp->post(q[ub] != s);
}
