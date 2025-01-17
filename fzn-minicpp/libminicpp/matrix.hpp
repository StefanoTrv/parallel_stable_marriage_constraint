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

#ifndef __MATRIX_H
#define __MATRIX_H

#include <vector>
#include <array>
#include <functional>
#include "intvar.hpp"


template<class Container,class T,int a> class matrix ;
template <class Container,class T,int a,int d> class MSlice;

template<class T, int a>
using
Matrix  = matrix<std::vector<T>,T,a>;

template<class T, int a, int d>
using VMSlice = MSlice<std::vector<T>,T,a,d>;

template <class Container,class T,int a,int d> class MSlice {
   matrix<Container,T,a>* _mtx;
   int         _flat;
   MSlice(matrix<Container,T,a>* m,int idx) : _mtx(m),_flat(idx) {}
public:
   friend class matrix<Container,T,a>;
   friend class MSlice<Container,T,a,d+1>;
   MSlice<Container,T,a,d-1> operator[](int i);
   const MSlice<Container,T,a,d-1> operator[](int i) const;
   int size() const { return _mtx->_dim[a-d];}
};
template<class Container,class T,int a> class MSlice<Container,T,a,1> {
   matrix<Container,T,a>* _mtx;
   int        _flat;
   friend class MSlice<Container,T,a,2>;
   MSlice(matrix<Container,T,a>* m,int idx) : _mtx(m),_flat(idx) {}
public:
   friend class matrix<Container,T,a>;
   T& operator[](int i);
   const T& operator[](int i) const;
   int size() const { return _mtx->_dim[a-1];}
};

// ======================================================================
// Matrix definition

template <typename T>
using alloct = stl::StackAdapter<T,Storage::Ptr>;

inline int computeSize(int a,std::initializer_list<int> li)
{
   int ttl = 1;
   for(auto e : li)
      ttl *= e;
   return ttl;
}

template<class Container,class T,int a> class matrix {
   Container    _data;
   std::array<int,a>  _dim;
public:
   friend class MSlice<Container,T,a,a-1>;
   matrix(std::initializer_list<int> li) {
      std::copy(li.begin(),li.end(),_dim.begin());
      int ttl = 1;
      for(int i=0;i <a;i++)
         ttl *= _dim[i];
      _data.resize(ttl);
   }
   matrix(Storage::Ptr store,std::initializer_list<int> li)
   : _data(computeSize(a,li) , alloct<T>(store)) {
      std::copy(li.begin(),li.end(),_dim.begin());
   }
   const Container& flat() const { return _data;}
   const int size(int d) const { return _dim[d];}
   MSlice<Container,T,a,a-1> operator[](int i) { return MSlice<Container,T,a,a-1>(this,i);}
};

template <class Container,class T,int a,int d>
MSlice<Container,T,a,d-1> MSlice<Container,T,a,d>::operator[](int i) {
   return MSlice<Container,T,a,d-1>(_mtx,_flat * _mtx->_dim[a - d] + i);
}
template<class Container,class T,int a>
T& MSlice<Container,T,a,1>::operator[](int i) {
   return _mtx->_data[_flat * _mtx->_dim[a - 1] + i];
}
template <class Container,class T,int a,int d>
const MSlice<Container,T,a,d-1> MSlice<Container,T,a,d>::operator[](int i) const {
   return MSlice<Container,T,a,d-1>(_mtx,_flat * _mtx->_dim[a - d] + i);
}
template<class Container,class T,int a>
const T& MSlice<Container,T,a,1>::operator[](int i) const {
   return _mtx->_data[_flat * _mtx->_dim[a - 1] + i];
}

template <class T,typename Body> std::vector<T> slice(int l,int u,Body b) {
   std::vector<T> x;
   for(int k=l;k < u;k++)
      x.emplace_back(b(k));
   return x;
}

#endif
