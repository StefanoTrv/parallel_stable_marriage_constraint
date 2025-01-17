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
 * StaticBitSet Implementation: Tim Curry
 */

#ifndef __BITSET_H
#define __BITSET_H

#include "trailable.hpp"
#include "handle.hpp"
#include "store.hpp"
#include <vector>

class StaticBitSet {
   std::vector<int>            _words;
   int                         _sz;
public:
   StaticBitSet() {}
   StaticBitSet(int sz);
   StaticBitSet(StaticBitSet&& bs) : _words(std::move(bs._words)),_sz(bs._sz) {}
   int operator[] (int i) { return _words[i];}
   void remove (int pos);
   bool contains(int pos);
};

class SparseBitSet {
   std::vector<trail<int>>     _words;  // length = nbWords
   std::vector<int>            _index;  // length = nbWords
   std::vector<int>            _mask;   // length = nbWords
   trail<int>                  _limit;
   int                         _sz;
   int                         _nbWords;
public:
   SparseBitSet(Trailer::Ptr eng, Storage::Ptr store, int sz);
   bool isEmpty() { return _limit == -1;}
   void clearMask();
   void reverseMask();
   void addToMask(StaticBitSet& m);
   void intersectWithMask();
   int intersectIndex(StaticBitSet& m);
   trail<int>& operator[] (int i) { return _words[i];}
   int operator[] (int i) const { return _words[i].value();}
};

#endif
