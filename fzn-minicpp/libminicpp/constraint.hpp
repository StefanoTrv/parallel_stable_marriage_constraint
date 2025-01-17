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

#ifndef __CONSTRAINT_H
#define __CONSTRAINT_H

#include <set>
#include <array>
#include <algorithm>
#include <iomanip>
#include <stdint.h>
#include "matrix.hpp"
#include "intvar.hpp"
#include "acstr.hpp"
#include "matching.hpp"

class EQc : public Constraint { // x == c
   var<int>::Ptr _x;
   int           _c;
public:
   EQc(var<int>::Ptr x,int c) : Constraint(x->getSolver()),_x(x),_c(c) {}
   void post() override;
   void propagate() override;
};

class NEQc : public Constraint { // x != c
   var<int>::Ptr _x;
   int           _c;
public:
   NEQc(var<int>::Ptr x,int c) : Constraint(x->getSolver()),_x(x),_c(c) {}
   void post() override;
   void propagate() override;
};

class LEQc : public Constraint { // x <= c
    var<int>::Ptr _x;
    int           _c;
public:
    LEQc(var<int>::Ptr x,int c) : Constraint(x->getSolver()),_x(x),_c(c) {}
    void post() override;
    void propagate() override;
};

class EQBinBC : public Constraint { // x == y + c
   var<int>::Ptr _x,_y;
   int _c;
public:
   EQBinBC(var<int>::Ptr x,var<int>::Ptr y,int c)
      : Constraint(x->getSolver()),_x(x),_y(y),_c(c) {}
   void post() override;
};


class EQTernBC : public Constraint { // x == y + z
  var<int>::Ptr _x,_y,_z;
public:
   EQTernBC(var<int>::Ptr x,var<int>::Ptr y,var<int>::Ptr z)
       : Constraint(x->getSolver()),_x(x),_y(y),_z(z) {}
   void post() override;
};

class EQTernBCbool : public Constraint { // x == y + b
  var<int>::Ptr _x,_y;
  var<bool>::Ptr _b;
public:
  EQTernBCbool(var<int>::Ptr x,var<int>::Ptr y,var<bool>::Ptr b)
    : Constraint(x->getSolver()),_x(x),_y(y),_b(b) {}
  void post() override;
};

class NEQBinBC : public Constraint { // x != y + c
   var<int>::Ptr _x,_y;
   int _c;
   trailList<Constraint::Ptr>::revNode* hdl[2];
   void print(std::ostream& os) const override;
public:
   NEQBinBC(var<int>::Ptr x,var<int>::Ptr y,int c)
      : Constraint(x->getSolver()), _x(x),_y(y),_c(c) {}
   void post() override;
};

class NEQBinBCLight : public Constraint { // x != y + c
   var<int>::Ptr _x,_y;
   int _c;
   void print(std::ostream& os) const override;
public:
   NEQBinBCLight(var<int>::Ptr& x,var<int>::Ptr& y,int c=0)
      : Constraint(x->getSolver()), _x(x),_y(y),_c(c) {}
   void post() override;
   void propagate() override;
};

class EQBinDC : public Constraint { // x == y + c
   var<int>::Ptr _x,_y;
   int _c;
public:
   EQBinDC(var<int>::Ptr& x,var<int>::Ptr& y,int c)
      : Constraint(x->getSolver()), _x(x),_y(y),_c(c) {}
   void post() override;
};

class Conjunction :public Constraint { // z == x && y
   var<bool>::Ptr _z,_x,_y;
public:
   Conjunction(var<bool>::Ptr z,var<bool>::Ptr x,var<bool>::Ptr y)
      : Constraint(x->getSolver()),_z(z),_x(x),_y(y) {}
   void post() override;
   void propagate() override;
};

class LessOrEqual :public Constraint { // x <= y
   var<int>::Ptr _x,_y;
public:
   LessOrEqual(var<int>::Ptr x,var<int>::Ptr y)
      : Constraint(x->getSolver()),_x(x),_y(y) {}
   void post() override;
   void propagate() override;
};

class IsEqual : public Constraint { // b <=> x == c
   var<bool>::Ptr _b;
   var<int>::Ptr _x;
   int _c;
public:
   IsEqual(var<bool>::Ptr b,var<int>::Ptr x,int c)
      : Constraint(x->getSolver()),_b(b),_x(x),_c(c) {}
   void post() override;
   void propagate() override;
};

class IsMember : public Constraint { // b <=> x in S
   var<bool>::Ptr _b;
   var<int>::Ptr _x;
   std::set<int> _S;
public:
   IsMember(var<bool>::Ptr b,var<int>::Ptr x,std::set<int> S)
      : Constraint(x->getSolver()),_b(b),_x(x),_S(S) {}
   void post() override;
   void propagate() override;
};

class IsLessOrEqual : public Constraint { // b <=> x <= c
   var<bool>::Ptr _b;
   var<int>::Ptr _x;
   int _c;
public:
   IsLessOrEqual(var<bool>::Ptr b,var<int>::Ptr x,int c)
      : Constraint(x->getSolver()),_b(b),_x(x),_c(c) {}
   void post() override;
};


class Sum : public Constraint { // s = Sum({x0,...,xk})
   Factory::Veci _x;
   trail<int>    _nUnBounds;
   trail<int>    _sumBounds;
   unsigned int _n;
   std::vector<unsigned long> _unBounds;
public:
   template <class Vec> Sum(const Vec& x, var<int>::Ptr s)
       : Constraint(s->getSolver()),
         _x(x.size() + 1,Factory::alloci(s->getStore())), 
         _nUnBounds(s->getSolver()->getStateManager(),(int)x.size()+1),
         _sumBounds(s->getSolver()->getStateManager(),0),
         _n((int)x.size() + 1),
         _unBounds(_n)
    {
       int i = 0;
       for(auto& xi : x)
          _x[i++] = xi;
       _x[_n-1] = Factory::minus(s);
       for(typename Vec::size_type i=0;i < _n;i++)
          _unBounds[i] = i;
    }
   void post() override;
   void propagate() override;
};

class SumBool : public Constraint {
   Factory::Vecb _x;
   int           _c;
   trail<int>    _nbOne,_nbZero;
public:
   template <class Vec> SumBool(const Vec& x,int c)
      : Constraint(x[0]->getSolver()),
        _x(x.size(),Factory::allocb(x[0]->getStore())),
        _c(c),
        _nbOne(x[0]->getSolver()->getStateManager(),0),
        _nbZero(x[0]->getSolver()->getStateManager(),0)
   {
       int i = 0;
       for(auto& xi : x)
          _x[i++] = xi;      
   }
   void propagateIdx(int k);
   void post() override;
};

class Clause : public Constraint { // x0 OR x1 .... OR xn
   std::vector<var<bool>::Ptr> _x;
   trail<int> _wL,_wR;
public:
   Clause(const std::vector<var<bool>::Ptr>& x);
   void post() override { propagate();}
   void propagate() override;
};

class IsClause : public Constraint { // b <=> x0 OR .... OR xn
   var<bool>::Ptr _b;
   std::vector<var<bool>::Ptr> _x;
   std::vector<int> _unBounds;
   trail<int>      _nUnBounds;
   Clause::Ptr        _clause;
public:
   IsClause(var<bool>::Ptr b,const std::vector<var<bool>::Ptr>& x);
   void post() override;
   void propagate() override;
};

class AllDifferentBinary :public Constraint {
   Factory::Veci _x;
public:
   template <class Vec> AllDifferentBinary(const Vec& x)
      : Constraint(x[0]->getSolver()),
        _x(x.size(),Factory::alloci(x[0]->getStore()))
   {
      int i  = 0;
      for(auto& xi : x)
         _x[i++] = xi;
  }
   void post() override;
};

class AllDifferentAC : public Constraint {
        Factory::Veci    _x;
        MaximumMatching _mm;
        PGraph*         _rg;
        int* _match,*_varFor;
        int _minVal,_maxVal;
        int _nVar,_nVal,_nNodes;
        int updateRange();
        int valNode(int vid) const noexcept { return vid - _minVal + _nVar;}
    public:
        template <class Vec> AllDifferentAC(const Vec& x)
                : Constraint(x[0]->getSolver()),
                  _x(x.begin(),x.end(),Factory::alloci(x[0]->getStore())),
                  _mm(_x,x[0]->getStore()) {}
        ~AllDifferentAC() {}
        void post() override;
        void propagate() override;
};


class Circuit : public Constraint {
protected:
   Factory::Veci  _x;
   trail<int>* _dest;
   trail<int>* _orig;
   trail<int>* _lengthToDest;
   int const offset;
   void bind(int i);
   void setup(CPSolver::Ptr cp);
public:
   template <class Vec> Circuit(const Vec& x, int offset = 0)
      : Constraint(x.front()->getSolver()),
        _x(x.size(), Factory::alloci(x.front()->getStore())),
        offset(offset)
   {
        for(auto i = 0u; i < x.size(); i += 1)
        {
            _x[i] = x[i];
        }
        auto cp = _x[0]->getSolver();
        setup(cp);
   }
   void post() override;
protected:
   virtual void postAllDifferent();
};

class Minimize : public Objective {
   var<int>::Ptr _obj;
   int        _primal;
   void print(std::ostream& os) const;
public:
   Minimize(var<int>::Ptr& x);
   void tighten() override;
   int value() const override { return _obj->min();}
};

class Maximize : public Objective {
   var<int>::Ptr _obj;
   int        _primal;
   void print(std::ostream& os) const;
public:
   Maximize(var<int>::Ptr& x);
   void tighten() override;
   int value() const override { return _obj->max();}
};

class Element2D : public Constraint {
   struct Triplet {
      int x,y,z;
      Triplet() : x(0),y(0),z(0) {}
      Triplet(int a,int b,int c) : x(a),y(b),z(c) {}
      Triplet(const Triplet& t) : x(t.x),y(t.y),z(t.z) {}
   };
   Matrix<int,2> _matrix;
   var<int>::Ptr _x,_y,_z;
   int _n,_m;
   trail<int>* _nRowsSup;
   trail<int>* _nColsSup;
   trail<int> _low,_up;
   std::vector<Triplet> _xyz;
   void updateSupport(int lostPos);
public:
   Element2D(const Matrix<int,2>& mat,var<int>::Ptr x,var<int>::Ptr y,var<int>::Ptr z);
   void post() override;;
   void propagate() override;
   void print(std::ostream& os) const override;
};

class Element1D : public Constraint {
   std::vector<int> _t;
   var<int>::Ptr _y;
   var<int>::Ptr _z;
public:
   Element1D(const std::vector<int>& array,var<int>::Ptr y,var<int>::Ptr z);
   void post() override;
};

class Element1DBasic : public Constraint { // z == t[y]
   std::vector<int> _t;
   var<int>::Ptr _y;
   var<int>::Ptr _z;
   struct Pair {
      int _k;
      int _v;
      Pair() : _k(0),_v(0) {}
      Pair(int k,int v) : _k(k),_v(v) {}
      Pair(const Pair& p) : _k(p._k),_v(p._k) {}
   };
   Pair* _kv;
   trail<int> _from,_to;
public:
   Element1DBasic(const std::vector<int>& array,var<int>::Ptr y,var<int>::Ptr z);
   void post() override;;
   void propagate() override;
   void print(std::ostream& os) const override;
};

class Element1DDC : public Constraint { // z =DC= t[y] (Domain consistent version)
        std::vector<int> _t;
        var<int>::Ptr    _y;
        var<int>::Ptr    _z;
        // Internal state
        struct DCIndex {    // for each value v in D(z) we maintain the support and the head of a list
            int        _v;   // the actual value from D(z)
            int        _k;   // the first index in t s.t. t[k]==v. The next index is in _list[k] (i.e., t[list[k]]==v)
            trail<int> _s;   // |{j in D(y) : t[j] == v }|
        };
        int      _endOfList; // endoflist marker
        DCIndex*    _values; // an array that holds, for each value in D(z), the support structure
        int*          _list; // holds _all_ the list indices (one list per index value) |_list| = range(t)
        int            _nbv; // number of values in _values
        BitDomain::Ptr _zOld,_yOld;
        int findIndex(int target) const;
        void zLostValue(int v);
        void yLostValue(int v);
    public:
        Element1DDC(const std::vector<int>& array,var<int>::Ptr y,var<int>::Ptr z);
        void post() override;
        void propagate() override;
        void print(std::ostream& os) const override;
};

class Element1DVar : public Constraint {  // _z = _array[y]
   Factory::Veci  _array;
   var<int>::Ptr   _y,_z;
   std::vector<int> _yValues;
   var<int>::Ptr _supMin,_supMax;
   int _zMin,_zMax;
   void equalityPropagate();
   void filterY();
public:
   template <class Vec> Element1DVar(const Vec& array,var<int>::Ptr y,var<int>::Ptr z)
      : Constraint(y->getSolver()),
        _array(array.size(),Factory::alloci(z->getStore())),
        _y(y),
        _z(z),
        _yValues(_y->size())
   {
      for(auto i = 0u;i < array.size();i++)
         _array[i] = array[i];
   }
   void post() override;
   void propagate() override;
};

namespace Factory
{
   inline Constraint::Ptr equal(var<int>::Ptr x,var<int>::Ptr y,int c=0) {
      return new (x->getSolver()) EQBinBC(x,y,c);
   }
   inline Constraint::Ptr equal(var<int>::Ptr x,var<int>::Ptr y,var<int>::Ptr z) {
      return new (x->getSolver()) EQTernBC(x,y,z);
   }
   inline Constraint::Ptr equal(var<int>::Ptr x,var<int>::Ptr y,var<bool>::Ptr b) {
     return new (x->getSolver()) EQTernBCbool(x,y,b);
   }
   inline Constraint::Ptr notEqual(var<int>::Ptr x,var<int>::Ptr y,int c=0) {
      return new (x->getSolver()) NEQBinBC(x,y,c);
   }
   inline Constraint::Ptr operator==(var<int>::Ptr x,const int c) {
      auto cp = x->getSolver();
      x->assign(c);
      cp->fixpoint();
      return nullptr;
   }
   inline Constraint::Ptr operator!=(var<int>::Ptr x, const int c) {
      auto cp = x->getSolver();
      x->remove(c);
      cp->fixpoint();
      return nullptr;
   }
   inline Constraint::Ptr inside(var<int>::Ptr x,std::set<int> S) {
      auto cp = x->getSolver();
      for(int v = x->min();v <= x->max();++v) {
         if (!x->contains(v)) continue;
         if (S.find(v) == S.end())
            x->remove(v);
      }
      cp->fixpoint();
      return nullptr;
   }
   inline Constraint::Ptr outside(var<int>::Ptr x,std::set<int> S) {
      auto cp = x->getSolver();
      for(int v : S) {
         if (x->contains(v))
            x->remove(v);
      }
      cp->fixpoint();
      return nullptr;
   }
   inline Constraint::Ptr operator==(var<bool>::Ptr x,const bool c) {
      return new (x->getSolver()) EQc((var<int>::Ptr)x,c);
   }
   inline Constraint::Ptr operator==(var<bool>::Ptr x,const int c) {
      return new (x->getSolver()) EQc((var<int>::Ptr)x,c);
   }
   inline Constraint::Ptr operator==(var<bool>::Ptr x,var<int>::Ptr y) {
      return new (x->getSolver()) EQBinBC(x,y,0);
   }
   inline Constraint::Ptr operator!=(var<bool>::Ptr x,const bool c) {
      return new (x->getSolver()) NEQc((var<int>::Ptr)x,c);
   }
   inline Constraint::Ptr operator!=(var<bool>::Ptr x,const int c) {
      return new (x->getSolver()) NEQc((var<int>::Ptr)x,c);
   }
   inline Constraint::Ptr operator!=(var<int>::Ptr x,var<int>::Ptr y) {
      return Factory::notEqual(x,y,0);
   }
   inline Constraint::Ptr operator<=(var<int>::Ptr x,var<int>::Ptr y) {
      return new (x->getSolver()) LessOrEqual(x,y);
   }
   inline Constraint::Ptr operator>=(var<int>::Ptr x,var<int>::Ptr y) {
      return new (x->getSolver()) LessOrEqual(y,x);
   }
   inline Constraint::Ptr operator<(var<int>::Ptr x,var<int>::Ptr y) {
      return new (x->getSolver()) LessOrEqual(x,y-1);
   }
   inline Constraint::Ptr operator>(var<int>::Ptr x,var<int>::Ptr y) {
      return new (x->getSolver()) LessOrEqual(y,x-1);
   }
   inline Constraint::Ptr operator<=(var<int>::Ptr x,const int c) {
      auto cp = x->getSolver();
      x->removeAbove(c);
      cp->fixpoint();
      return nullptr;
   }
   inline Constraint::Ptr operator>=(var<int>::Ptr x,const int c) {
      auto cp = x->getSolver();
      x->removeBelow(c);
      cp->fixpoint();
      return nullptr;
   }
    inline Constraint::Ptr operator<=(var<bool>::Ptr x,const int c) {
        x->removeAbove(c);
        x->getSolver()->fixpoint();
        return nullptr;
    }
    inline Constraint::Ptr operator>=(var<bool>::Ptr x,const int c) {
        x->removeBelow(c);
        x->getSolver()->fixpoint();
        return nullptr;
    }
   inline Objective::Ptr minimize(var<int>::Ptr x) {
      return new Minimize(x);
   }
   inline Objective::Ptr maximize(var<int>::Ptr x) {
      return new Maximize(x);
   }
   inline var<int>::Ptr operator+(var<int>::Ptr x,var<int>::Ptr y) { // x + y
      int min = x->min() + y->min();
      int max = x->max() + y->max();
      var<int>::Ptr z = makeIntVar(x->getSolver(),min,max);
      x->getSolver()->post(equal(z,x,y));
      return z;
   }
   inline var<int>::Ptr operator-(var<int>::Ptr x,var<int>::Ptr y) { // x - y
      int min = x->min() - y->max();
      int max = x->max() - y->max();
      var<int>::Ptr z = makeIntVar(x->getSolver(),min,max);
      x->getSolver()->post(equal(x,z,y));
      return z;
   }
   inline var<bool>::Ptr operator*(var<bool>::Ptr x,var<bool>::Ptr y) { // x * y (bool) meaning x && y
      var<bool>::Ptr z = makeBoolVar(x->getSolver());
      x->getSolver()->post(new (x->getSolver()) Conjunction(z,x,y));
      return z;
   }
   inline var<bool>::Ptr operator&&(var<bool>::Ptr x,var<bool>::Ptr y) { // x * y (bool) meaning x && y
      var<bool>::Ptr z = makeBoolVar(x->getSolver());
      x->getSolver()->post(new (x->getSolver()) Conjunction(z,x,y));
      return z;
   }
   inline var<bool>::Ptr isEqual(var<int>::Ptr x,const int c) {
      var<bool>::Ptr b = makeBoolVar(x->getSolver());
      TRYFAIL
         x->getSolver()->post(new (x->getSolver()) IsEqual(b,x,c));
      ONFAIL
      ENDFAIL
      return b;
   }
   inline Constraint::Ptr isMember(var<bool>::Ptr b, var<int>::Ptr x, const std::set<int> S) {
     return new (x->getSolver()) IsMember(b,x,S);
   }
   inline var<bool>::Ptr isMember(var<int>::Ptr x,const std::set<int> S) {
      var<bool>::Ptr b = makeBoolVar(x->getSolver());
      TRYFAIL
         x->getSolver()->post(new (x->getSolver()) IsMember(b,x,S));
      ONFAIL
      ENDFAIL
      return b;
   }
   inline var<bool>::Ptr isLessOrEqual(var<int>::Ptr x,const int c) {
      var<bool>::Ptr b = makeBoolVar(x->getSolver());
      TRYFAIL
         x->getSolver()->post(new (x->getSolver()) IsLessOrEqual(b,x,c));
      ONFAIL
      ENDFAIL
      return b;
   }
   inline var<bool>::Ptr isLess(var<int>::Ptr x,const int c) {
      return isLessOrEqual(x,c - 1);
   }
   inline var<bool>::Ptr isLargerOrEqual(var<int>::Ptr x,const int c) {
      return isLessOrEqual(- x,- c);
   }
   inline var<bool>::Ptr isLarger(var<int>::Ptr x,const int c) {
      return isLargerOrEqual(x , c + 1);
   }
   template <class Vec> var<int>::Ptr sum(Vec& xs) {
      int sumMin = 0,sumMax = 0;
      for(const auto& x : xs) {
         sumMin += x->min();
         sumMax += x->max();
      }
      auto cp = xs[0]->getSolver();
      auto s = Factory::makeIntVar(cp,sumMin,sumMax);
      cp->post(new (cp) Sum(xs,s));
      return s;
   }
   template <class Vec> var<int>::Ptr sum(CPSolver::Ptr cp,Vec& xs) {
      int sumMin = 0,sumMax = 0;
      for(const auto& x : xs) {
         sumMin += x->min();
         sumMax += x->max();
      }
      auto s = Factory::makeIntVar(cp,sumMin,sumMax);
      if (xs.size() > 0)
         cp->post(new (cp) Sum(xs,s));
      return s;
   }
   template <class Vec> Constraint::Ptr sum(const Vec& xs,var<int>::Ptr s) {
      return new (xs[0]->getSolver()) Sum(xs,s);
   }
   inline Constraint::Ptr sum(const Factory::Veci& xs,int s) {
      auto sv = Factory::makeIntVar(xs[0]->getSolver(),s,s);
      return new (xs[0]->getSolver()) Sum(xs,sv);
   }
   inline Constraint::Ptr sum(const std::vector<var<int>::Ptr>& xs,int s) {
      auto sv = Factory::makeIntVar(xs[0]->getSolver(),s,s);
      return new (xs[0]->getSolver()) Sum(xs,sv);
   }
   inline Constraint::Ptr sum(const Factory::Vecb& xs,int s) {
      return new (xs[0]->getSolver()) SumBool(xs,s);
   }
   inline Constraint::Ptr sum(const std::vector<var<bool>::Ptr>& xs,int s) {
      return new (xs[0]->getSolver()) SumBool(xs,s);
   }
   template <class Vec> Constraint::Ptr clause(const Vec& xs) {
      return new (xs[0]->getSolver()) Clause(xs);
   }
   template <class Vec> Constraint::Ptr isClause(var<bool>::Ptr b,const Vec& xs) {
      return new (b->getSolver()) IsClause(b,xs);
   }
   inline var<bool>::Ptr implies(var<bool>::Ptr a,var<bool>::Ptr b) { // a=>b is not(a) or b is (1-a)+b >= 1
      std::vector<var<int>::Ptr> left = {1- (var<int>::Ptr)a,b};
      return isLargerOrEqual(sum(left),1);
   }
   template <class Vec> Constraint::Ptr allDifferent(const Vec& xs) {
      return new (xs[0]->getSolver()) AllDifferentBinary(xs);
   }
   template <class Vec> Constraint::Ptr allDifferentAC(const Vec& xs) {
      return new (xs[0]->getSolver()) AllDifferentAC(xs);
   }
   template <class Vec>  Constraint::Ptr circuit(const Vec& xs) {
      return new (xs[0]->getSolver()) Circuit(xs);
   }
   template <class Vec> Constraint::Ptr element(const Vec& array,var<int>::Ptr y,var<int>::Ptr z) {
      std::vector<int> flat(array.size());
      for(int i=0;i < (int)array.size();i++)
         flat[i] = array[i];
      return new (y->getSolver()) Element1D(flat,y,z);
   }
   template <class Vec> Constraint::Ptr elementVar(const Vec& xs,var<int>::Ptr y,var<int>::Ptr z) {
       std::vector<var<int>::Ptr> flat(xs.size());
       for(int i=0;i<xs.size();i++)
           flat[i] = xs[i];
       return new (y->getSolver()) Element1DVar(flat,y,z);
   }
   inline var<int>::Ptr element(Matrix<int,2>& d,var<int>::Ptr x,var<int>::Ptr y) {
      int min = INT32_MAX,max = INT32_MIN;
      for(int i=0;i<d.size(0);i++)
         for(int j=0;j < d.size(1);j++) {
            min = min < d[i][j] ? min : d[i][j];
            max = max > d[i][j] ? max : d[i][j];
         }
      auto z = makeIntVar(x->getSolver(),min,max);
      x->getSolver()->post(new (x->getSolver()) Element2D(d,x,y,z));
      return z;
   }
   inline Constraint::Ptr element(const VMSlice<int,2,1>& array,var<int>::Ptr y,var<int>::Ptr z) {
      std::vector<int> flat(array.size());
      for(int i=0;i < array.size();i++)
         flat[i] = array[i];
      return new (y->getSolver()) Element1D(flat,y,z);
   }
   template <class Vec> inline var<int>::Ptr element(const Vec& array,var<int>::Ptr y) {
      int min = INT32_MAX,max = INT32_MIN;
      std::vector<int> flat(array.size());
      for(auto i=0u;i < array.size();i++) {
         const int v = flat[i] = array[i];
         min = min < v ? min : v;
         max = max > v ? max : v;
      }
      auto z = makeIntVar(y->getSolver(),min,max);
      y->getSolver()->post(new (y->getSolver()) Element1D(flat,y,z));
      return z;
   }
   template <class Vec> var<int>::Ptr elementVar(const Vec& xs,var<int>::Ptr y) {
      int min = INT32_MAX,max = INT32_MIN;
      std::vector<var<int>::Ptr> flat(xs.size());
      for(auto i=0u;i < xs.size();i++) {
         const auto& v = flat[i] = xs[i];
         min = min < v->min() ? min : v->min();
         max = max > v->max() ? max : v->max();
      }
      auto z = makeIntVar(y->getSolver(),min,max);
      y->getSolver()->post(new (y->getSolver()) Element1DVar(flat,y,z));
      return z;
   }
};

void printCstr(Constraint::Ptr c);

#endif
