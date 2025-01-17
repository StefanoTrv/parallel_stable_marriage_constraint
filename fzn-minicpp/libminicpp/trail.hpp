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

#ifndef __TRAIL_H
#define __TRAIL_H

#include <memory>
#include <stack>
#include <tuple>
#include "state.hpp"

class Entry {
public:
   virtual void relocate(long) noexcept {}
   virtual void restore() noexcept = 0;
};

class Trailer :public StateManager {
   std::stack<Entry*>      _trail;
   std::stack<std::tuple<int,std::size_t,long>>  _tops;
   mutable int             _magic;
   long  _lastNode;
   char* _block;
   std::size_t  _bsz;
   std::size_t  _btop;
   bool      _enabled;
public:
   Trailer();
   ~Trailer();
   void enable()  override { _enabled = true;}
   void trail(Entry* e) { if (e) _trail.push(e);}
   typedef handle_ptr<Trailer> Ptr;
   void resize();
   int magic() const { return _magic;}
   void incMagic() { _magic++;}
   long push();
   void pop();
   void popToNode(long node);
   void clear();

   void saveState() override;
   void restoreState() override;
   void withNewState(const std::function<void(void)>& body) override;

   friend void* operator new(std::size_t sz,Trailer::Ptr& e) noexcept;
   friend void* operator new[](std::size_t sz,Trailer::Ptr& e) noexcept;
};

inline void* operator new[](std::size_t sz,Trailer::Ptr& e) noexcept 
{
   return operator new(sz,e);
}

inline void* operator new(std::size_t sz,Trailer::Ptr& e) noexcept
{
   if (e->_enabled) {
      char* ptr = e->_block + e->_btop;
      e->_btop += sz;
      if (e->_btop >= e->_bsz) {
         e->_btop -= sz;
         e->resize();
         ptr = e->_block + e->_btop;
         e->_btop += sz;
      }
      return ptr;
   } else return nullptr;
}


#endif
