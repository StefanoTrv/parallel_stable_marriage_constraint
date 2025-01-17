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

#ifndef __STORE_H
#define __STORE_H

#include <vector>
#include "handle.hpp"
#include "trail.hpp"
#include "trailable.hpp"
#include "stlAllocAdapter.hpp"

#define SEGSIZE (1 << 22)

class Storage {
   struct Segment {
      char*      _base;
      std::size_t  _sz;
      Segment(std::size_t tsz);
      ~Segment();
      typedef std::shared_ptr<Segment> Ptr;
   };
   Trailer::Ptr                         _ctx;
   std::vector<Storage::Segment::Ptr> _store;
   const std::size_t   _segSize;
   trail<size_t>           _top;   
   trail<unsigned>         _seg;
public:
   Storage(Trailer::Ptr ctx,std::size_t defSize = SEGSIZE); 
   ~Storage();
   typedef handle_ptr<Storage> Ptr;
   void* allocate(std::size_t sz);
   void free(void* ptr) {}
   std::size_t capacity() const;
   std::size_t usage() const;
};

inline void* operator new(std::size_t sz,Storage::Ptr store)
{
   return store->allocate(sz);
}

inline void* operator new[](std::size_t sz,Storage::Ptr store)
{
   return store->allocate(sz);
}

class Pool;
class PoolMark {
   friend class Pool;
   size_t   _top;
   unsigned _seg;
   PoolMark(size_t top,unsigned seg) : _top(top),_seg(seg) {}
};

class Pool {
   struct Segment {
      char*      _base;
      std::size_t  _sz;
      Segment(std::size_t tsz) { _base = new char[tsz];_sz = tsz;}
      ~Segment()               { delete[] _base;}
      typedef std::shared_ptr<Segment> Ptr;
   };
   Segment** _store;
   const std::size_t   _segSize;
   size_t                  _top;
   unsigned                _seg;
   unsigned              _nbSeg;
   unsigned                _mxs;
public:
   Pool(std::size_t defSize = SEGSIZE);
   ~Pool();
   typedef handle_ptr<Pool> Ptr;
   void* allocate(std::size_t sz);
   void free(void* ptr) {}
   void clear() { _top = 0;_seg = 0;}
   void clear(const PoolMark& m) { _top = m._top;_seg = m._seg;}
   PoolMark mark() const noexcept { return {_top,_seg};}
   std::size_t capacity() const noexcept { return _segSize;}
   std::size_t usage() const noexcept { return (_nbSeg - 1) * _segSize  + _top;}
};

inline void* operator new(std::size_t sz,Pool::Ptr store)
{
   return store->allocate(sz);
}

inline void* operator new[](std::size_t sz,Pool::Ptr store)
{
   return store->allocate(sz);
}

#endif
