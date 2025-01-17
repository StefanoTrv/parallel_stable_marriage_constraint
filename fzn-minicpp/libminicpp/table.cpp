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
 * 
 * TableCT Implementation: Tim Curry
 */

#include "table.hpp"
#include <string.h>

void TableCT::post()
{
   propagate();
   for (const auto& vp : _vars)
      vp->propagateOnDomainChange(this); // for each variable, propagate on change to domain
}

void TableCT::propagate()  // enforceGAC
{
   // calculate Sval
   int varIndex;
   _Sval.clear();
   _Ssup.clear();
   for (const auto& vp : _vars) {
      varIndex = _entries.at(vp->getId()).getIndex();
      if (vp->size() != _lastSizes[varIndex]) {
         _Sval.push_back(vp);
         _lastSizes[varIndex] = vp->size();  // update lastSizes for vars in Sval
      }
      // calculate Ssup
      if (vp->size() > 1)
         _Ssup.push_back(vp);
   }
   // call updateTable
   updateTable();
   // check size of currTable
   if (_currTable.isEmpty())
      failNow();
   // call filterDomains
   filterDomains();
}

void TableCT::filterDomains()
{
   int varIndex, valIndex, wIndex, iMin;
   for (const auto& vp : _Ssup) {
      varIndex = _entries.at(vp->getId()).getIndex();
      iMin = _entries.at(vp->getId()).getMin();
      for (int val = vp->min(); val < vp->max() + 1; val++) {
         valIndex = val - iMin;
         wIndex = _residues[varIndex][valIndex];
         if (wIndex == -1) {
            vp->remove(val);
         }
         else {
            if ((_currTable[wIndex] & _supports[varIndex][valIndex][wIndex]) == 0) {
               wIndex = _currTable.intersectIndex(_supports[varIndex][valIndex]);
               if (wIndex != -1) {
                  _residues[varIndex][valIndex] = wIndex;
               }
               else {
                  vp->remove(val);
               }
            }
         }
      }
      _lastSizes[varIndex] = vp->size();
   }
}

void TableCT::updateTable()
{
   int varIndex, valIndex, iMin;
   for (const auto& vp : _Sval) {
      varIndex = _entries.at(vp->getId()).getIndex();
      iMin = _entries.at(vp->getId()).getMin();
      _currTable.clearMask();
      for (int val = vp->min(); val < vp->max() + 1; val++) {
         valIndex = val - iMin;
         if (vp->contains(val))
            _currTable.addToMask(_supports[varIndex][valIndex]);
      }
      _currTable.intersectWithMask();
      if (_currTable.isEmpty())
         break;
   }
}
