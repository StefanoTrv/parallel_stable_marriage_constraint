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

#ifndef __REGULAR_H
#define __REGULAR_H

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

struct Transition {
   int from;
   int symbol;
   int to;
};

class Automaton {
   CPSolver::Ptr _cp;
   Range<int>   _smb; // first to last symbol range (alphabet for the FSA)
   Range<int>   _stt; // first to last state range (set of state identifiers in FSA)
   int         _start;      // start state
   std::set<int> _finals;   // all final states in the FSA
   std::vector<Transition> _t; // transition table for the FSA
public:
   typedef handle_ptr<Automaton> Ptr;
   Automaton(CPSolver::Ptr s,int f,int l,int fState,int lState,
             int start,std::set<int> finals,const std::vector<Transition>& tf)
      : _cp(s),_smb(f,l),_stt(fState,lState),
        _start(start),_finals(finals),_t(tf)
   {}
   CPSolver::Ptr getSolver() { return _cp;}
   auto getStates() const { return _stt;}
   auto getSymbols()const { return _smb;}
   auto getStart()  const { return _start;}
   auto getFinals() const {return _finals;}
   std::vector<std::vector<int>> transition() const;
};

class Regular : public Constraint {
   Automaton::Ptr _A;
   Factory::Veci  _x;
public:
   template <class Vec> Regular(const Vec& x,Automaton::Ptr A)
      : Constraint(x[0]->getSolver()),_A(A),
        _x(x.size(),Factory::alloci(x[0]->getStore()))
   {
      int i  = 0;
      for(auto& xi : x)
         _x[i++] = xi;
   }
   void post() override;
};

namespace Factory {
   inline Automaton::Ptr automaton(CPSolver::Ptr cp,int f,int l,int fs,int ls,
                                   int start,
                                   std::set<int> finals,
                                   const std::vector<Transition>& tf)
   {
      return new (cp) Automaton(cp,f,l,fs,ls,start,finals,tf);
   }
   template <class Vec> Constraint::Ptr regular(const Vec& x,Automaton::Ptr aPtr) {
      return new (aPtr->getSolver()) Regular(x,aPtr);
   }
}

#endif
