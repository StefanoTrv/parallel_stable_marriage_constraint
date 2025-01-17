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

#ifndef __INTVAR_H
#define __INTVAR_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <functional>
#include <assert.h>
#include "avar.hpp"
#include "varitf.hpp"
#include "acstr.hpp"
#include "solver.hpp"
#include "domain.hpp"
#include "trailList.hpp"
#include "matrix.hpp"

class IntVarImpl : public var<int> {
   CPSolver::Ptr           _solver;
   BitDomain::Ptr             _dom;
   trailList<Constraint::Ptr> _onBindList;
   trailList<Constraint::Ptr> _onBoundsList;
   trailList<Constraint::Ptr> _onDomList;
   struct DomainListener :public IntNotifier {
      IntVarImpl* theVar;
      DomainListener(IntVarImpl* x) : theVar(x) {}
      void empty() override;
      void bind() override;
      void change()  override;
      void changeMin() override;
      void changeMax() override;
   };
   DomainListener*       _domListener;
public:
   IntVarImpl(CPSolver::Ptr& cps,int min,int max);
   IntVarImpl(CPSolver::Ptr& cps,int n) : IntVarImpl(cps,0,n-1) {}
   Storage::Ptr getStore() override   { return _solver->getStore();}
   CPSolver::Ptr getSolver() override { return _solver;}
   int min() const override { return _dom->min();}
   int max() const override { return _dom->max();}
   int size() const override { return _dom->size();}
   bool isBound() const override { return _dom->isBound();}
   bool contains(int v) const override { return _dom->member(v);}
   bool containsBase(int v) const override { return _dom->memberBase(v);}
   int getIthVal(int index) const noexcept {assert(0 < index and index <= _dom->size()); return _dom->getIthVal(index);}
   void dump(int min, int max, unsigned int * dump) const override { _dom->dump(min,max,dump);};
   bool changed() const noexcept override  { return _dom->changed();}
   bool changedMin() const noexcept override  { return _dom->changedMin();}
   bool changedMax() const noexcept override  { return _dom->changedMax();}

   void assign(int v) override;
   void remove(int v) override;
   void removeBelow(int newMin) override;
   void removeAbove(int newMax) override;
   void updateBounds(int newMin,int newMax) override;
   
   TLCNode* whenBind(std::function<void(void)>&& f) override;
   TLCNode* whenBoundsChange(std::function<void(void)>&& f) override;
   TLCNode* whenDomainChange(std::function<void(void)>&& f) override;
   TLCNode* propagateOnBind(Constraint::Ptr c)  override         { return _onBindList.emplace_back(std::move(c));}
   TLCNode* propagateOnBoundChange(Constraint::Ptr c)  override  { return _onBoundsList.emplace_back(std::move(c));}
   TLCNode* propagateOnDomainChange(Constraint::Ptr c ) override { return _onDomList.emplace_back(std::move(c));}

    std::ostream& print(std::ostream& os) const override {
       // if (size() == 1)
       //     os << min();
       // else
            os << "x_" << getId() << '(' << *_dom << ')';
        return os;
    }
};

class IntVarViewOpposite :public var<int> {
   var<int>::Ptr _x;
public:
   IntVarViewOpposite(const var<int>::Ptr& x) : _x(x) {}
   Storage::Ptr getStore() override   { return _x->getStore();}
   CPSolver::Ptr getSolver() override { return _x->getSolver();}
   int min() const  override { return - _x->max();}
   int max() const  override { return - _x->min();}
   int size() const override { return _x->size();}
   bool isBound() const override { return _x->isBound();}
   bool contains(int v) const override { return _x->contains(-v);}
   bool changed() const noexcept override  { return _x->changed();}
   bool changedMin() const noexcept override  { return _x->changedMax();}
   bool changedMax() const noexcept override  { return _x->changedMin();}
   
   void assign(int v) override { _x->assign(-v);}
   void remove(int v) override { _x->remove(-v);}
   void removeBelow(int newMin) override { _x->removeAbove(-newMin);}
   void removeAbove(int newMax) override { _x->removeBelow(-newMax);}
   void updateBounds(int newMin,int newMax) override { _x->updateBounds(-newMax,-newMin);}
   
   TLCNode* whenBind(std::function<void(void)>&& f) override { return _x->whenBind(std::move(f));}
   TLCNode* whenBoundsChange(std::function<void(void)>&& f) override { return _x->whenBoundsChange(std::move(f));}
   TLCNode* whenDomainChange(std::function<void(void)>&& f) override { return _x->whenDomainChange(std::move(f));}
   TLCNode* propagateOnBind(Constraint::Ptr c)          override { return _x->propagateOnBind(c);}
   TLCNode* propagateOnBoundChange(Constraint::Ptr c)   override { return _x->propagateOnBoundChange(c);}
   TLCNode* propagateOnDomainChange(Constraint::Ptr c ) override { return _x->propagateOnDomainChange(c);}
   std::ostream& print(std::ostream& os) const override {
      os << '{';
      for(int i = min();i <= max() - 1;i++) 
         if (contains(i)) os << i << ',';
      if (size()>0) os << max();
      return os << '}';      
   }
};

static inline int floorDiv(int a,int b) {
   const int q = a/b;
   return (a < 0 && q * b != a) ? q - 1 : q;
}
static inline int ceilDiv(int a,int b) {
   const int q = a / b;
   return (a > 0 && q * b != a) ? q + 1 : q;
}


class IntVarViewMul : public var<int> {
    var<int>::Ptr _x;
    int _a;
public:
   IntVarViewMul(const var<int>::Ptr& x,int a) : _x(x),_a(a) { assert(a> 0);}
   Storage::Ptr getStore() override   { return _x->getStore();}
   CPSolver::Ptr getSolver() override { return _x->getSolver();}
   int min() const  override { return _a * _x->min();}
   int max() const  override { return _a * _x->max();}
   int size() const override { return _x->size();}
   bool isBound() const override { return _x->isBound();}
   bool contains(int v) const override { return (v % _a != 0) ? false : _x->contains(v / _a);}
   bool changed() const noexcept override  { return _x->changed();}
   bool changedMin() const noexcept override  { return _x->changedMin();}
   bool changedMax() const noexcept override  { return _x->changedMax();}

   void assign(int v) override {
      if (v % _a == 0)
         _x->assign(v / _a);
      else failNow();
   }
   void remove(int v) override {
      if (v % _a == 0)
         _x->remove(v / _a);
   }
   void removeBelow(int v) override { _x->removeBelow(ceilDiv(v,_a));}
   void removeAbove(int v) override { _x->removeAbove(floorDiv(v,_a));}
   void updateBounds(int min,int max) override { _x->updateBounds(ceilDiv(min,_a),floorDiv(max,_a));}   
   TLCNode* whenBind(std::function<void(void)>&& f) override { return _x->whenBind(std::move(f));}
   TLCNode* whenBoundsChange(std::function<void(void)>&& f) override { return _x->whenBoundsChange(std::move(f));}
   TLCNode* whenDomainChange(std::function<void(void)>&& f) override { return _x->whenDomainChange(std::move(f));}
   TLCNode* propagateOnBind(Constraint::Ptr c)          override { return _x->propagateOnBind(c);}
   TLCNode* propagateOnBoundChange(Constraint::Ptr c)   override { return _x->propagateOnBoundChange(c);}
   TLCNode* propagateOnDomainChange(Constraint::Ptr c ) override { return _x->propagateOnDomainChange(c);}
   std::ostream& print(std::ostream& os) const override {
      os << '{';
      for(int i = min();i <= max() - 1;i++) 
         if (contains(i)) os << i << ',';
      if (size()>0) os << max();
      return os << '}';      
   }
};

class IntVarViewOffset : public var<int> {
    var<int>::Ptr _x;
    int _o;
public:
   IntVarViewOffset(const var<int>::Ptr& x,int o) : _x(x),_o(o) {}
   Storage::Ptr getStore() override   { return _x->getStore();}
   CPSolver::Ptr getSolver() override { return _x->getSolver();}
   int min() const  override { return _o + _x->min();}
   int max() const  override { return _o + _x->max();}
   int size() const override { return _x->size();}
   bool isBound() const override { return _x->isBound();}
   bool contains(int v) const override { return _x->contains(v - _o);}
   bool changed() const noexcept override  { return _x->changed();}
   bool changedMin() const noexcept override  { return _x->changedMin();}
   bool changedMax() const noexcept override  { return _x->changedMax();}

   void assign(int v) override { _x->assign(v - _o);}
   void remove(int v) override { _x->remove(v - _o);}
   void removeBelow(int v) override { _x->removeBelow(v - _o);}
   void removeAbove(int v) override { _x->removeAbove(v - _o);}
   void updateBounds(int min,int max) override { _x->updateBounds(min - _o,max - _o);}   
   TLCNode* whenBind(std::function<void(void)>&& f) override { return _x->whenBind(std::move(f));}
   TLCNode* whenBoundsChange(std::function<void(void)>&& f) override { return _x->whenBoundsChange(std::move(f));}
   TLCNode* whenDomainChange(std::function<void(void)>&& f) override { return _x->whenDomainChange(std::move(f));}
   TLCNode* propagateOnBind(Constraint::Ptr c)          override { return _x->propagateOnBind(c);}
   TLCNode* propagateOnBoundChange(Constraint::Ptr c)   override { return _x->propagateOnBoundChange(c);}
   TLCNode* propagateOnDomainChange(Constraint::Ptr c ) override { return _x->propagateOnDomainChange(c);}
   std::ostream& print(std::ostream& os) const override {
      os << '{';
      for(int i = min();i <= max() - 1;i++) 
         if (contains(i)) os << i << ',';
      if (size()>0) os << max();
      return os << '}';      
   }
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif

template <>
class var<bool> : public IntVarImpl {
public:
    typedef handle_ptr<var<bool>> Ptr;
    var<bool>(CPSolver::Ptr& cps) : IntVarImpl(cps,0,1) {}
    bool isTrue() const { return min()==1;}
    bool isFalse() const { return max()==0;}
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif
void assign(bool b)  { IntVarImpl::assign(b);}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
};
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

inline std::ostream& operator<<(std::ostream& os,const var<int>::Ptr& xp) {
    return xp->print(os);
}
inline std::ostream& operator<<(std::ostream& os,const var<bool>::Ptr& xp) {
    return xp->print(os);
}

template <class T,class A> inline std::ostream& operator<<(std::ostream& os,const std::vector<T,A>& v) {
   os << '[';
   for(typename std::vector<T,A>::size_type i = 0; i < v.size(); ++i) {
      os << v[i];
      if (i != v.size() - 1)
            os << ", ";
   }
   return os << ']';
}

namespace Factory {
   template <class Vec> var<int>::Ptr elementVar(const Vec& xs,var<int>::Ptr y);
};

template <class T,class A = std::allocator<T>>
class EVec : public std::vector<T,A> {
   typedef std::vector<T,A> BVec;
public:
   using BVec::BVec;
   //EVec() = delete;
   var<int>::Ptr operator[](var<int>::Ptr i) {
      return Factory::elementVar(*this,i);
   }
   T& operator[](typename BVec::size_type i)      { return BVec::operator[](i);}
   T operator[](typename BVec::size_type i) const { return BVec::operator[](i);}
};

namespace Factory {
   using alloci = stl::StackAdapter<var<int>::Ptr,Storage::Ptr>;
   using allocb = stl::StackAdapter<var<bool>::Ptr,Storage::Ptr>;
   using Veci   = EVec<var<int>::Ptr,alloci>;
   using Vecb   = EVec<var<bool>::Ptr,allocb>;
   var<int>::Ptr makeIntVar(CPSolver::Ptr cps,int min,int max);
   var<int>::Ptr makeIntVar(CPSolver::Ptr cps,std::initializer_list<int> vals);
   var<int>::Ptr makeIntVar(CPSolver::Ptr cps,std::vector<int> const & vals);
   var<bool>::Ptr makeBoolVar(CPSolver::Ptr cps);
   var<bool>::Ptr makeBoolVar(CPSolver::Ptr cps, bool value);
   inline var<int>::Ptr minus(var<int>::Ptr x)     { return new (x->getSolver()) IntVarViewOpposite(x);}
   inline var<int>::Ptr operator-(var<int>::Ptr x) { return minus(x);}
   inline var<int>::Ptr operator*(var<int>::Ptr x,int a) {
      if (a == 0)
         return makeIntVar(x->getSolver(),0,0);
      else if (a==1)
         return x;
      else if (a <0)
         return minus(new (x->getSolver()) IntVarViewMul(x,-a));
      else return new (x->getSolver()) IntVarViewMul(x,a);
   }
   inline var<int>::Ptr operator*(int a,var<int>::Ptr x) { return x * a;}
   inline var<int>::Ptr operator*(var<bool>::Ptr x,int a)  { return Factory::operator*((var<int>::Ptr)x,a);}
   inline var<int>::Ptr operator*(int a,var<bool>::Ptr x)  { return x * a;}
   inline var<int>::Ptr operator+(var<int>::Ptr x,int a) { return new (x->getSolver()) IntVarViewOffset(x,a);}
   inline var<int>::Ptr operator+(int a,var<int>::Ptr x) { return new (x->getSolver()) IntVarViewOffset(x,a);}
   inline var<int>::Ptr operator-(var<int>::Ptr x,const int a) { return new (x->getSolver()) IntVarViewOffset(x,-a);}
   inline var<int>::Ptr operator-(const int a,var<int>::Ptr x) { return new (x->getSolver()) IntVarViewOffset(-x,a);}
   Veci intVarArray(CPSolver::Ptr cps,int sz,int min,int max);
   Veci intVarArray(CPSolver::Ptr cps,int sz,int n);
   Veci intVarArray(CPSolver::Ptr cps,int sz);
   Vecb boolVarArray(CPSolver::Ptr cps,int sz,bool createVar = true);
   template<typename Fun> Veci intVarArray(CPSolver::Ptr cps,int sz,Fun body) {
      auto x = intVarArray(cps,sz);
      for(auto i=0u;i < x.size();i++)
         x[i] = body(i);
      return x;
   }
   template <class Vec>             size_t count(Vec& v) { return v.size();}
   template <class Vec,class ...Ts> size_t count(Vec& v,Ts... vecs) {
      return v.size() + count(vecs...);
   }
   template <class Vec> int fill(Vec& dest,int from,Vec& src) {
      for(auto& v : src)
         dest[from++]=v;
      return from;
   }
   template <class Vec,class ...Ts> int fill(Vec& dest,int from,Vec& src,Ts... vecs) {
      for(auto& v : src)
         dest[from++]=v;
      return fill(dest,from,vecs...);
   }
   template <class Vec,class ...Ts> Vec collect(Vec& v,Ts... vecs) {
      auto nbv = count(v,vecs...);
      auto cps = (*v.cbegin())->getSolver();
      Vec rv = Factory::intVarArray(cps,nbv);
      fill(rv,0,v,vecs...);
      return rv;
   }
};

void printVar(var<int>* x);
void printVar(var<int>::Ptr x);

#endif
