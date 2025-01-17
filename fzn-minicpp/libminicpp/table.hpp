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

#ifndef __TABLE_H
#define __TABLE_H

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

class TableCT : public Constraint {
   class Entry {
      const int _index;
      const int _min;
      const int _max;
   public:
      Entry(int index, int min, int max) :
         _index(index),
         _min(min),
         _max(max)
      {}
      int getIndex() { return _index;}
      int getMin() { return _min;}
      int getMax() {return _max;}
   };
   std::map<int, Entry>                                _entries;
   std::vector<std::vector<StaticBitSet>>              _supports;
   std::vector<std::vector<int>>                       _table;
   std::vector<std::vector<int>>                       _residues;
   SparseBitSet                                        _currTable;
   std::vector<var<int>::Ptr>                          _vars;
   std::vector<var<int>::Ptr>                          _Sval;
   std::vector<var<int>::Ptr>                          _Ssup;
   std::vector<trail<int>>                             _lastSizes;
   void                                                filterDomains();
   void                                                updateTable();
public:
   template <class Vec> TableCT(const Vec& x, const std::vector<std::vector<int>>& table)
      : Constraint(x[0]->getSolver()),
        _table(table),
        _currTable(x[0]->getSolver()->getStateManager(), x[0]->getStore(), table.size())  // build SparseBitSet for vectors in table
   {
      int currIndex = 0;
      for (const auto& vp : x) {
         _vars.push_back(vp);
         _entries.emplace(vp->getId(), Entry(currIndex, vp->min(), vp->max()));  // build entries for each var
         _lastSizes.push_back(trail<int>(x[0]->getSolver()->getStateManager(), vp->size()));  // record initial var domain size
         _supports.emplace_back(std::vector<StaticBitSet>(0));  // init vector of bitsets for var's supports
         _residues.emplace_back(std::vector<int>());
         std::vector<StaticBitSet>& v = _supports.back();
         std::vector<int>& r = _residues.back();
         for (int val=vp->min(); val < vp->max()+1; val++) {
            v.emplace_back(StaticBitSet((int)table.size()));  // make bitset for this value
            // loop through table to see which table vectors support this value
            for (auto i=0u; i < table.size(); i++) {
               const std::vector<int>& tableEntry = table[i];
               if (tableEntry[currIndex] != val)
                  v.back().remove(i);  // this tableEntry is not a support for val
            }
            r.push_back(_currTable.intersectIndex(v.back()));
         }
         currIndex++;
      }
   }
   ~TableCT() {}
   void post() override;
   void propagate() override;
};

namespace Factory {
   template <class Vec, class T> Constraint::Ptr table(const Vec& x, const T& table) {
      return new (x[0]->getSolver()) TableCT(x, table);
   }
};

#endif
