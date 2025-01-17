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

#ifndef __TRAILABLE_H
#define __TRAILABLE_H

#include "trail.hpp"

template<class T>
class TrailEntry: public Entry {
        T*  _at;
        T  _old;
    public:
        TrailEntry(T* at) : _at(at), _old(*at) {}
        void restore() noexcept { *_at = _old;}
};

template<class T> class trail {
   Trailer::Ptr       _ctx;
   int              _magic;
   T                _value;
   void save(int nm) {
      _magic = nm;
      Entry* entry = new (_ctx) TrailEntry(&_value);
      _ctx->trail(entry);
   }
public:
   trail() : _ctx(nullptr),_magic(-1),_value(T())  {}
   trail(Trailer::Ptr ctx,const T& v = T()) : _ctx(ctx),_magic(ctx->magic()),_value(v) {}
   bool fresh() const { return _magic != _ctx->magic();}
   Trailer::Ptr ctx() const { return _ctx;}
   operator T() const { return _value;}
   T value() const { return _value;}
   trail<T>& operator=(const T& v);
   trail<T>& operator+=(const T& v);
   trail<T>& operator-=(const T& v);
   trail<T>& operator++(); // pre-increment
   T operator++(int); // post-increment
};

template<class T>
trail<T>& trail<T>::operator=(const T& v)
{
   int cm = _ctx->magic();
   if (_magic != cm)
      save(cm);
   _value = v;
   return *this;
}

template <class T>
trail<T>& trail<T>::operator++() { // pre-increment
   int cm = _ctx->magic();
   if (_magic != cm)
      save(cm);
   _value += 1;
   return *this;
}

template <class T>
T trail<T>::operator++(int) { // post-increment
   T rv = _value;
   int cm = _ctx->magic();
   if (_magic != cm)
      save(cm);
   ++_value;
   return rv;
}


template<class T>
trail<T>& trail<T>::operator+=(const T& v) {
   int cm = _ctx->magic();
   if (_magic != cm)
      save(cm);
   _value += v;
   return *this;
}

template<class T>
trail<T>& trail<T>::operator-=(const T& v) {
   int cm = _ctx->magic();
   if (_magic != cm)
      save(cm);
   _value -= v;
   return *this;
}

#endif
