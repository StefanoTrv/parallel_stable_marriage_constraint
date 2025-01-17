//
//  utilities.hpp
//  mdd
//
//  Created by zitoun on 1/11/20.
//  Copyright © 2020 zitoun. All rights reserved.
//
#pragma once

#include <set>
#include <map>
#include <string.h>
#include <xmmintrin.h>
#include <cmath>
#include <ostream>

class ValueSet {
   char* _data;
   int  _min,_max;
   int  _sz;
public:
   ValueSet(const std::set<int>& s) {
      _min = *s.begin();
      _max = *s.begin();
      for(auto v : s) {
         _min = _min < v ? _min : v;
         _max = _max > v ? _max : v;
      }
      _sz = _max - _min + 1;
      _data = new char[_sz];
      memset(_data,0,sizeof(char)*_sz);
      for(auto v : s)
         _data[v - _min] = 1;
   }
   template <typename Container>
   ValueSet(const Container& s) {
      _min = (s.size()) ? s[0]->getId() : 0;
      _max = (s.size()) ? s[0]->getId() : -1;
      for(auto v : s) {
         _min = _min < v->getId() ? _min : v->getId();
         _max = _max > v->getId() ? _max : v->getId();
      }
      _sz = _max - _min + 1;
      _data = new char[_sz];
      memset(_data,0,sizeof(char)*_sz);
      for(auto v : s)
         _data[v->getId() - _min] = 1;
   }
   ValueSet(const ValueSet& s) : _min(s._min), _max(s._max), _sz(s._sz)
   {
      _data = new char[_sz];
      memcpy(_data,s._data,sizeof(char)*_sz);
   }
   bool member(int v) const noexcept {
      if (_min <= v && v <= _max)
         return _data[v - _min];
      else return false;
   }
   friend inline std::ostream& operator<<(std::ostream& os,const ValueSet& s)
   {
      os << "{" << s._min << "," << s._max << std::endl;
      for(int i = s._min; i <= s._max; i++)
         s.member(i);
      return os << "}" << std::endl;
   }
};

template <typename T>
class ValueMap {
   T* _data;
   int  _min,_max;
   int  _sz;
public:
   ValueMap(int min, int max, T defaut, const std::map<int,T>& s) {
      _min = min;
      _max = max;
      _sz = _max - _min + 1;
      _data = new T[_sz];
      memset(_data,defaut,sizeof(T)*_sz);
      for(auto kv : s)
         _data[kv.first - min] = kv.second;
   }
   ValueMap(int min, int max, std::function<T(int)>& clo){
      _min = min;
      _max = max;
      _sz = _max - _min + 1;
      _data = new T[_sz];
      for(int i = min; i <= max; i++)
         _data[i-_min] = clo(i);
   }
   const T& operator[](int idx) const{
      return _data[idx - _min];
   }
};

inline short propNbWords(short nb) { return (nb >> 6) + (((nb & 63) != 0) ? 1 : 0);}

class MDDPropSet {
   short    _mxw;
   short    _nbp;
   long long* _t;
public:
   MDDPropSet() : _mxw(0),_nbp(0),_t(nullptr) {}
   MDDPropSet(int nb) {
      _mxw = (nb >> 6) + (((nb & 63) != 0) ? 1 : 0);
      _nbp = nb;
      _t   = new long long[_mxw];
      for(int i=0;i<_mxw;i++) _t[i]=0;
   }
   MDDPropSet(long long* buf,int nb) {
      _mxw = (nb >> 6) + (((nb & 63) != 0) ? 1 : 0);
      _nbp = nb;
      _t   = buf;
      for(int i=0;i<_mxw;i++) _t[i]=0;
   }
   short nbWords() const noexcept { return _mxw;}
   short nbProps() const noexcept { return _nbp;}
   void clear() noexcept {
      for(short i=0;i < _mxw;i++)
         _t[i] = 0;
   }
   short size() const noexcept {
      short ttl = 0;
      for(short i=0;i < _mxw;++i)
         ttl += __builtin_popcountl(_t[i]);
      return ttl;
   }
   void setProp(int p) noexcept       { _t[p >> 6] |= (1ull << (p & 63));}
   bool hasProp(int p) const noexcept { return (_t[p >> 6] &  (1ull << (p & 63))) != 0;}
   void unionWith(const MDDPropSet& ps) noexcept {
      switch (_mxw) {
         case 1: _t[0] |= ps._t[0];break;
#if defined(__x86_64__)
         case 2: {
            __m128i op0 = *(__m128i*)_t;
            __m128i op1 = *(__m128i*)ps._t;
            *(__m128i*)_t = _mm_or_si128(op0,op1);
         }break;
         case 3: {
            __m128i op0 = *(__m128i*)_t;
            __m128i op1 = *(__m128i*)ps._t;
            *(__m128i*)_t = _mm_or_si128(op0,op1);
            _t[2] |= ps._t[2];
         }break;
#endif
         default: {
            for(short i=0;i < _mxw;++i)
               _t[i] |= ps._t[i]; 
         }
      }
   }
   void interWith(const MDDPropSet& ps) noexcept {
      for(short i=0;i < _mxw;i++)
         _t[i] &= ps._t[i];
   }
   class iterator : public std::iterator<std::input_iterator_tag,short,short> {
      long long* _t;
      const short _nbw;
      short _cwi;    // current word index
      long long _cw; // current word
      iterator(long long* t,short nbw,short at)
         : _t(t),_nbw(nbw),_cwi(at),_cw((at < nbw) ? t[at] : 0) {
         while (_cw == 0 && ++_cwi < _nbw) 
            _cw = _t[_cwi];         
      }
      iterator(long long* t,short nbw) : _t(t),_nbw(nbw),_cwi(nbw),_cw(0) {} // end constructor
   public:
      iterator& operator++()  noexcept {
         long long test = _cw & -_cw;  // only leaves LSB at 1
         _cw ^= test;                  // clear LSB
         while (_cw == 0 && ++_cwi < _nbw)  // all bits at zero-> done with this word.            
            _cw = _t[_cwi];        
         return *this;
      }
      iterator operator++(int) { iterator retval = *this; ++(*this); return retval;}
      bool operator==(iterator other) const {return _cwi == other._cwi && _cw == other._cw;}
      bool operator!=(iterator other) const {return !(*this == other);}
      short operator*() const   { return (_cwi<<6) + __builtin_ctzl(_cw);}
      friend class MDDPropSet;
   };
   iterator begin() const { return iterator(_t,_mxw,0);}
   iterator end()   const { return iterator(_t,_mxw);}
   friend std::ostream& operator<<(std::ostream& os,const MDDPropSet& ps) {
      os << '[' << ps.size() << ']' << '{';
      for(short i : ps) os << i << ' ';
      return os << '}';
   }
};