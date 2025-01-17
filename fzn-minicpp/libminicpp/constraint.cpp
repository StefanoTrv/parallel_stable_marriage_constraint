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

#include "constraint.hpp"
#include <string.h>

void printCstr(Constraint::Ptr c) {
   c->print(std::cout);
   std::cout << std::endl;
}

void EQc::post()
{
   propagate();
}

void EQc::propagate()
{
    _x->assign(_c);
}

void NEQc::post()
{
    propagate();
}

void NEQc::propagate()
{
   _x->remove(_c);
}

void LEQc::post()
{
    propagate();
}

void LEQc::propagate()
{
    _x->removeAbove(_c);
}

void EQBinBC::post()
{
   if (_x->isBound())
      _y->assign(_x->min() - _c);
   else if (_y->isBound())
      _x->assign(_y->min() + _c);
   else {
      _x->updateBounds(_y->min() + _c,_y->max() + _c);
      _y->updateBounds(_x->min() - _c,_x->max() - _c);
      _x->whenBoundsChange([this] {
         _y->updateBounds(_x->min() - _c,_x->max() - _c);
      });
      _y->whenBoundsChange([this] {
         _x->updateBounds(_y->min() + _c,_y->max() + _c);
      });
   }
}

void EQTernBC::post()
{
   // x == y + z
   if (_x->isBound() && _y->isBound())
      _z->assign(_x->min() - _y->min());
   else if (_x->isBound() && _z->isBound())
      _y->assign(_x->min() - _z->min());
   else if (_y->isBound() && _z->isBound())
      _x->assign(_y->min() + _z->min());
   else {
      _x->updateBounds(_y->min() + _z->min(),_y->max() + _z->max());
      _y->updateBounds(_x->min() - _z->max(),_x->max() - _z->min());
      _z->updateBounds(_x->min() - _y->max(),_x->max() - _y->min());
      _x->whenBoundsChange([this] {
         _y->updateBounds(_x->min() - _z->max(),_x->max() - _z->min());
         _z->updateBounds(_x->min() - _y->max(),_x->max() - _y->min());
      });
      _y->whenBoundsChange([this] {
         _x->updateBounds(_y->min() + _z->min(),_y->max() + _z->max());
         _z->updateBounds(_x->min() - _y->max(),_x->max() - _y->min());
      });
      _z->whenBoundsChange([this] {
         _x->updateBounds(_y->min() + _z->min(),_y->max() + _z->max());
         _y->updateBounds(_x->min() - _z->max(),_x->max() - _z->min());
      });
   }
}

void EQTernBCbool::post()
{
   // x == y + b
   if (_x->isBound() && _y->isBound()) {
      if (_x->min() - _y->min() == 1) {
         _b->assign(true);
      }
      else if (_x->min() - _y->min() == 0) {
         _b->assign(false);
      } else {
         failNow();
      }
   }
   else if (_x->isBound() && _b->isTrue())
      _y->assign(_x->min() - 1);
   else if (_x->isBound() && _b->isFalse())
      _y->assign(_x->min());
   else if (_y->isBound() && _b->isTrue())
      _x->assign(_y->min() + 1);
   else if (_y->isBound() && _b->isFalse())
      _x->assign(_y->min());
   else {
      _x->updateBounds(_y->min() + _b->min(),_y->max() + _b->max());
      _y->updateBounds(_x->min() - _b->max(),_x->max() - _b->min());
      _b->updateBounds(_x->min() - _y->max(),_x->max() - _y->min());
      
      _x->whenBoundsChange([this] {
         _y->updateBounds(_x->min() - _b->max(),_x->max() - _b->min());
         _b->updateBounds(_x->min() - _y->max(),_x->max() - _y->min());
      });
      _y->whenBoundsChange([this] {
         _x->updateBounds(_y->min() + _b->min(),_y->max() + _b->max());
         _b->updateBounds(_x->min() - _y->max(),_x->max() - _y->min());
      });
      _b->whenBoundsChange([this] {
         _x->updateBounds(_y->min() + _b->min(),_y->max() + _b->max());
         _y->updateBounds(_x->min() - _b->max(),_x->max() - _b->min());
      });
   }
}

void NEQBinBC::print(std::ostream& os) const
{
   os << _x << " != " << _y << " + " << _c << std::endl;
}

void NEQBinBC::post()
{
   if (_x->isBound())
      _y->remove(_x->min() - _c);
   else if (_y->isBound())
      _x->remove(_y->min() + _c);
   else {
      hdl[0] = _x->whenBind([this] {
         _y->remove(_x->min() - _c);
         hdl[0]->detach();
         hdl[1]->detach();
      });
      hdl[1] = _y->whenBind([this] {
         _x->remove(_y->min() + _c);
         hdl[0]->detach();
         hdl[1]->detach();
      });
   }
}

void NEQBinBCLight::print(std::ostream& os) const
{
   os << _x << " !=L " << _y << " + " << _c << std::endl;
}
void NEQBinBCLight::post()
{
   if (_y->isBound())
      _x->remove(_y->min() + _c);
   else if (_x->isBound())
      _y->remove(_x->min() - _c);
   else {
      _x->propagateOnBind(this);
      _y->propagateOnBind(this);
   }
}
void NEQBinBCLight::propagate()
{
   if (_y->isBound())
      _x->remove(_y->min() + _c);
   else
      _y->remove(_x->min() - _c);
   setActive(false);
}

void EQBinDC::post()
{
   if (_x->isBound())
      _y->assign(_x->min() - _c);
   else if (_y->isBound())
      _x->assign(_y->min() + _c);
   else {
      _x->updateBounds(_y->min() + _c,_y->max() + _c);
      _y->updateBounds(_x->min() - _c,_x->max() - _c);
      int lx = _x->min(), ux = _x->max();
      for(int i = lx ; i <= ux; i++)
         if (!_x->contains(i))
            _y->remove(i - _c);
      int ly = _y->min(),uy = _y->max();
      for(int i= ly;i <= uy; i++)
         if (!_y->contains(i))
            _x->remove(i + _c);
      _x->whenBind([this] { _y->assign(_x->min() - _c);});
      _y->whenBind([this] { _x->assign(_y->min() + _c);});
   }
}

void Conjunction::post() // z == x && y
{
   propagate();
   if (!_x->isBound()) _x->propagateOnBind(this);
   if (!_y->isBound()) _y->propagateOnBind(this);
   if (!_z->isBound()) _z->propagateOnBind(this);
}

void Conjunction::propagate()
{
   if (_z->isBound()) {
      if (_z->isTrue()) {
         setActive(false);
         _x->assign(true);
         _y->assign(true);
      } else {
         if (_x->isTrue()) {
            setActive(false);
            _y->assign(false);
         } else if (_y->isTrue()) {
            setActive(false);
            _x->assign(false);
         }
      }
   } else {
      if (_x->isBound() && _y->isBound()) {
         setActive(false);
         _z->assign(_x->min() && _y->min());
      } else if (_x->isFalse() || _y->isFalse()) {
         setActive(false);
         _z->assign(false);
      }        
   }
}

void LessOrEqual::post()
{
   _x->propagateOnBoundChange(this);
   _y->propagateOnBoundChange(this);
   propagate();
}
    
void LessOrEqual::propagate()
{
   _x->removeAbove(_y->max());
   _y->removeBelow(_x->min());
   setActive(_x->max() >= _y->min());
}

Minimize::Minimize(var<int>::Ptr& x)
   : _obj(x),_primal(0x7FFFFFFF)
{
   auto tightening = std::function<void(void)>([&]()
   {
        TRYFAIL
            _obj->removeAbove(_primal);
        ONFAIL
            notifyFailure();
            failNow();
        ENDFAIL
   });
   _obj->getSolver()->onFixpoint(tightening);
}

void Minimize::print(std::ostream& os) const
{
   os << "minimize(" << *_obj << ", primal = " << _primal << ")";
}

void Minimize::tighten()
{
    //assert(_obj->isBound());
   _primal = _obj->max() - 1;
   notifyFailure();
   failNow();
}

Maximize::Maximize(var<int>::Ptr& x)
   : _obj(x),_primal(0x80000001)
{
   auto tightening = std::function<void(void)>([&]()
   {
        TRYFAIL
            _obj->removeBelow(_primal);
        ONFAIL
            notifyFailure();
            failNow();
        ENDFAIL
   });
   _obj->getSolver()->onFixpoint(tightening);
}

void Maximize::print(std::ostream& os) const
{
   os << "maximize(" << *_obj << ", primal = " << _primal << ")";
}

void Maximize::tighten()
{
   assert(_obj->isBound());
   _primal = _obj->min() + 1;
   notifyFailure();
   failNow();
}

void IsEqual::post() 
{
   propagate();
   if (isActive()) {
      _x->propagateOnDomainChange(this);
      _b->propagateOnBind(this);
   }
}

void IsEqual::propagate() 
{
   if (_b->isTrue()) {
      _x->assign(_c);
      setActive(false);
   } else if (_b->isFalse()) {
      _x->remove(_c);
      setActive(false);
   } else if (!_x->contains(_c)) {
      _b->assign(false);
      setActive(false);
   } else if (_x->isBound()) {
      _b->assign(true);
      setActive(false);
   }
}

void IsMember::post() 
{
   propagate();
   if (isActive()) {
      _x->propagateOnDomainChange(this);
      _b->propagateOnBind(this);
   }
}

void IsMember::propagate() 
{  
   if (_b->isTrue()) {
      int xMin = _x->min(), xMax = _x->max();
      for (int v=xMin; v<=xMax; v++) {
         // if v is not in S: remove from domain of x
         if (_x->contains(v) && (_S.find(v) == _S.end()))
            _x->remove(v);
      }
      setActive(false);
   } else if (_b->isFalse()) {
      // remove all elements in S from domain of x
      for (std::set<int>::iterator it=_S.begin(); it!=_S.end(); ++it) {
         _x->remove(*it);
      }
      setActive(false);
   } else if (_x->isBound()) {
      int v = _x->min();
      if (_S.find(v)!=_S.end())
         _b->assign(true);
      else
         _b->assign(false);
      setActive(false);
   } else {
      // both b and x are not bound: check if x still has value in S and a value not in S
      bool hasMemberInS = false;
      bool hasMemberOutS = false;

      int xMin = _x->min(), xMax = _x->max();
      for (int v=xMin; v<=xMax; v++) {
         if (_x->contains(v)) {
            if (_S.find(v) == _S.end()) {
               hasMemberOutS = true;
            }
            else {
               hasMemberInS = true;
            }
         }
         if ((hasMemberInS == true) && (hasMemberOutS == true))
            break;
      }
      if (hasMemberInS==false) {
         _b->assign(false);
         setActive(false);
      }
      else if (hasMemberOutS==false) {
         _b->assign(true);
         setActive(false);
      }
   }
}


void IsLessOrEqual::post()
{
   if (_b->isTrue())
      _x->removeAbove(_c);
   else if (_b->isFalse())
      _x->removeBelow(_c + 1);
   else if (_x->max() <= _c)
      _b->assign(1);
   else if (_x->min() > _c)
      _b->assign(0);
   else {
      _b->whenBind([b=_b,x=_x,c=_c] {
         if (b->isTrue())
            x->removeAbove(c);
         else x->removeBelow(c + 1);
      });
      _x->whenBoundsChange([b=_b,x=_x,c=_c] {
         if (x->max() <= c)
            b->assign(1);
         else if (x->min() > c)
            b->assign(0);
      });        
   }
}

void Sum::post()
{
   for(auto& var : _x)
      var->propagateOnBoundChange(this);
   propagate();
}

void Sum::propagate()
{  
   int nU = _nUnBounds;
   int sumMin = _sumBounds,sumMax = _sumBounds;
   for(int i = nU - 1; i >= 0;i--) {
      auto idx = _unBounds[i];
      sumMin += _x[idx]->min();
      sumMax += _x[idx]->max();
      if (_x[idx]->isBound()) {
         _sumBounds = _sumBounds + _x[idx]->min();
         _unBounds[i] = _unBounds[nU - 1];
         _unBounds[nU - 1] = idx;
         nU--;
      }
   }
   _nUnBounds = nU;
   if (0 < sumMin ||  sumMax < 0)
      failNow();
   for(int i = nU - 1; i >= 0;i--) {
      auto idx = _unBounds[i];
      _x[idx]->removeAbove(-(sumMin - _x[idx]->min()));
      _x[idx]->removeBelow(-(sumMax - _x[idx]->max()));
   }
}

void SumBool::post() 
{
   int nbTrue = 0,nbPos = 0;
   for(auto i=0u;i < _x.size();i++) {
      nbTrue += _x[i]->min() == 1;
      nbPos  += !_x[i]->isBound();
   }
   if (nbTrue > _c)
      failNow();
   if (nbTrue == _c) {
      for(auto xi : _x)
         if (!xi->isBound())
            xi->assign(false);
      return ;
   }
   if (nbTrue + nbPos < _c)
      failNow();
   if (nbTrue + nbPos == _c) {
      for(auto xi : _x)
         if (!xi->isBound())
            xi->assign(true);
      return ;
   }
   _nbOne = nbTrue;
   _nbZero = (int)_x.size() - nbTrue - nbPos;
   for(auto k=0u;k < _x.size();k++) {
      if (_x[k]->isBound()) continue;
      _x[k]->whenBind([this,k]() {
         this->propagateIdx(k);
      });
   }
}

void SumBool::propagateIdx(int k)
{
   int nb1 = 0;
   if (_x[k]->isTrue()) {
      if (_nbOne + 1 == _c) {
         for(auto i=0u;i < _x.size();i++) {
            nb1 += _x[i]->isTrue();
            if (!_x[i]->isBound())
               _x[i]->assign(false);
         }
         if (nb1 != _c)
            failNow();
      } else
         _nbOne = _nbOne + 1;
   } else {
      if ((int)_x.size() - _nbZero - 1 == _c) {
         for(auto i=0u;i < _x.size();i++) {
            nb1 += _x[i]->min();
            if (!_x[i]->isBound()) {
               _x[i]->assign(true);
               ++nb1; 
            }
         }
         if (nb1 != _c)
            failNow();
      } else
         _nbZero = _nbZero + 1;
   }
}

Clause::Clause(const std::vector<var<bool>::Ptr>& x)
   : Constraint(x[0]->getSolver()),
     _wL(x[0]->getSolver()->getStateManager(),0),
     _wR(x[0]->getSolver()->getStateManager(),(int)x.size() - 1)
{
   for(auto xi : x) _x.push_back(xi);
}

void Clause::propagate()
{
   const long n = _x.size();
   int i = _wL;
   while (i < n && _x[i]->isBound()) {
      if (_x[i]->isTrue()) {
         setActive(false);
         return;
      }
      i += 1;
   }
   _wL = i;
   i = _wR;
   while (i>=0 && _x[i]->isBound()) {
      if (_x[i]->isTrue()) {
         setActive(false);
         return;
      }
      i -= 1;
   }
   _wR = i;
   if (_wL > _wR) failNow();
   else if (_wL == _wR) {
      _x[_wL]->assign(true);
      setActive(false);
   } else {
      assert(_wL != _wR);
      assert(!_x[_wL]->isBound());
      assert(!_x[_wR]->isBound());
      _x[_wL]->propagateOnBind(this);
      _x[_wR]->propagateOnBind(this);
   }
}

IsClause::IsClause(var<bool>::Ptr b,const std::vector<var<bool>::Ptr>& x)
   : Constraint(x[0]->getSolver()),
     _b(b),
     _unBounds(x.size()),
     _nUnBounds(x[0]->getSolver()->getStateManager(),(int)x.size())
{
   for(auto xi : x) _x.push_back(xi);
   for(auto i = 0u; i < _x.size();i++)
      _unBounds[i] = i;
   _clause = new (x[0]->getSolver()) Clause(x);
}

void IsClause::post()
{
   _b->propagateOnBind(this);
   for(auto& xi : _x)
      xi->propagateOnBind(this);
}

void IsClause::propagate()
{
   auto cp = _x[0]->getSolver();
   if (_b->isTrue()) {
      setActive(false);
      cp->post(_clause,false);
   } else if (_b->isFalse()) {
      for(auto& xi : _x)
         xi->assign(false);
      setActive(false);
   } else {
      int nU = _nUnBounds;
      for(int i = nU - 1;i >=0;i--) {
         int idx = _unBounds[i];
         auto y = _x[idx];
         if (y->isBound()) {
            if (y->isTrue()) {
               _b->assign(true);
               setActive(false);
               return;
            }
            _unBounds[i] = _unBounds[nU -1];
            _unBounds[nU - 1] = idx;
            nU--;
         }
      }
      if (nU == 0) {
         _b->assign(false);
         setActive(false);
      }
      _nUnBounds = nU;
   }
}

void AllDifferentBinary::post()
{
   CPSolver::Ptr cp = _x[0]->getSolver();
   const long n = _x.size();
   for(int i=0;i < n;i++) 
      for(int j=i+1;j < n;j++)
         cp->post(new (cp) NEQBinBCLight(_x[i],_x[j]));    
}

void AllDifferentAC::post()
{
    CPSolver::Ptr cp = _x[0]->getSolver();
    _nVar    = (int)_x.size();
    _nVal    = updateRange();
    _mm.setup();
    for(int i=0;i < _nVar;i++)
        _x[i]->propagateOnDomainChange(this);
    _match   = new (cp) int[_nVar];
    _varFor  = new (cp) int[_nVal];
    _nNodes  = _nVar + _nVal + 1;
    _rg = new (cp) PGraph(_nVar,_minVal,_maxVal,_match,_varFor,_x.data());
    propagate();
}

int AllDifferentAC::updateRange()
{
    _minVal = INT32_MAX;
    _maxVal = INT32_MIN;
    Factory::Veci::pointer x = _x.data();
    for(int i=0;i < _nVar;i++) {
        _minVal = std::min(_minVal,x[i]->min());
        _maxVal = std::max(_maxVal,x[i]->max());
    }
    return _maxVal - _minVal + 1;
}

void AllDifferentAC::propagate()
{
    int size = _mm.compute(_match);
    if (size < _nVar)
        failNow();
    updateRange();
    _rg->setLiveValues(_minVal,_maxVal);

    for(int i=0;i < _nVal;++i) _varFor[i] = -1;
    for(int i=0;i < _nVar;++i) _varFor[_match[i] - _minVal] = i;

    //std::cout << "DD:" << _nbd << "    " << _nbmd << "   " << _nVar << "/" << (_maxVal - _minVal + 1) << "\n";

    int nc = 0;
    int* scc = (int*)alloca(sizeof(int)*_nNodes);
    _rg->SCC([&scc,&nc](int n,int nd[]) {
        for(int i=0;i < n;i++)
            scc[nd[i]] = nc;
        ++nc;
    });
    if (nc > 1) {
        Factory::Veci::pointer x = _x.data();
        for(int i=0;i < _nVar;i++) {
            const int ub = x[i]->max();
            for(int v = x[i]->min(); v <= ub;++v)
                if (_match[i] != v && scc[i] != scc[valNode(v)] && x[i]->containsBase(v))
                    x[i]->remove(v);
        }
    }
}

void Circuit::setup(CPSolver::Ptr cp)
{
   _dest = new (cp) trail<int>[_x.size()];
   _orig = new (cp) trail<int>[_x.size()];
   _lengthToDest = new (cp) trail<int>[_x.size()];
   for (auto i = 0u; i < _x.size(); i++)
   {
        new (_dest+i) trail<int>(cp->getStateManager(),i);
        new (_orig+i) trail<int>(cp->getStateManager(),i);
        new (_lengthToDest+i) trail<int>(cp->getStateManager(),0);
   }
}

void Circuit::post()
{
   postAllDifferent();

   if (_x.size() == 1) {
      _x[0]->assign( offset + 0);
      return;
   }
   for(auto i = 0u; i < _x.size(); i++)
   {
       _x[i]->remove(offset + i);
   }
   for (auto i = 0u; i < _x.size(); i++)
   {
        if (_x[i]->isBound())
            bind(i);
        else
         _x[i]->whenBind([i,this]() { bind(i);});
   }
}

void Circuit::postAllDifferent()
{
    auto cp = _x[0]->getSolver();
    cp->post(Factory::allDifferentAC(_x));
}

void Circuit::bind(int i)
{
   int j = _x[i]->min() - offset;
   int origi = _orig[i];
   int destj = _dest[j];
   _dest[origi] = destj;
   _orig[destj] = origi;
   int length = _lengthToDest[origi] + _lengthToDest[j] + 1;
   _lengthToDest[origi] = length;
   if (length < (int)_x.size() - 1)
      _x[destj]->remove(offset + origi);
}

Element2D::Element2D(const Matrix<int,2>& mat,var<int>::Ptr x,var<int>::Ptr y,var<int>::Ptr z)
   : Constraint(x->getSolver()),
     _matrix(mat),
     _x(x),_y(y),_z(z),
     _n(mat.size(0)),
     _m(mat.size(1)),
     _low(x->getSolver()->getStateManager(),0),
     _up(x->getSolver()->getStateManager(),_n * _m - 1)
{
   for(int i=0;i < _matrix.size(0);i++)
      for(int j=0;j < _matrix.size(1);j++)
         _xyz.push_back(Triplet(i,j,_matrix[i][j]));
   std::sort(_xyz.begin(),_xyz.end(),[](const Triplet& a,const Triplet& b) {
      return a.z < b.z;
   });
   _nColsSup = new (x->getSolver()) trail<int>[_n];
   _nRowsSup = new (x->getSolver()) trail<int>[_m];
   auto sm = x->getSolver()->getStateManager();
   for(int i=0;i<_n;i++)
      new (_nColsSup + i) trail<int>(sm,_m);
   for(int j=0;j <_m;j++)
      new (_nRowsSup + j) trail<int>(sm,_n);
}

void Element2D::updateSupport(int lostPos)
{
   int nv1 = _nColsSup[_xyz[lostPos].x] = _nColsSup[_xyz[lostPos].x] - 1;
   if (nv1 == 0)
      _x->remove(_xyz[lostPos].x);
   int nv2 = _nRowsSup[_xyz[lostPos].y] = _nRowsSup[_xyz[lostPos].y] - 1;
   if (nv2==0)
      _y->remove(_xyz[lostPos].y);
}


void Element2D::post()
{
   _x->updateBounds(0,_n-1);
   _y->updateBounds(0,_m-1);
   _x->propagateOnDomainChange(this);
   _y->propagateOnDomainChange(this);
   _z->propagateOnBoundChange(this);
   propagate();
}

void Element2D::propagate()
{
   int l =  _low,u = _up;
   int zMin = _z->min(),zMax = _z->max();
   while (_xyz[l].z < zMin || !_x->contains(_xyz[l].x) || !_y->contains(_xyz[l].y)) {
      updateSupport(l++);
      if (l > u) failNow();
   }
   while (_xyz[u].z > zMax || !_x->contains(_xyz[u].x) || !_y->contains(_xyz[u].y)) {
      updateSupport(u--);
      if (l > u) failNow();
   }
   _z->updateBounds(_xyz[l].z,_xyz[u].z);
   _low = l;
   _up  = u;
}

void Element2D::print(std::ostream& os) const
{
   os << "element2D(" << _x << ',' << _y << ',' << _z << ')' << std::endl;
}

Element1DBasic::Element1DBasic(const std::vector<int>& array,var<int>::Ptr y,var<int>::Ptr z)
   : Constraint(y->getSolver()),_t(array),_y(y),_z(z),
     _from(_y->getSolver()->getStateManager(),-1),
     _to(_y->getSolver()->getStateManager(),-1) // z == array[y]
{
   _kv = NULL;
}

void Element1DBasic::post() 
{
   if (_y->isBound()) {
      _z->assign(_t[_y->min()]);
   } else if (_z->isBound()) {
      const int zv = _z->min();
      for(int v = _y->min(); v <= _y->max();v++) { // loop on index
         if (v < 0 || v >= (int)_t.size() || _t[v] != zv) // does not yield val in dom
            _y->remove(v);
      }
   } else { // nobody bound.
      _kv = new (_y->getSolver()) Pair[_t.size()];
      for(int k = 0;k < (int)_t.size();k++) 
         _kv[k] = {k,_t[k]};
      qsort(_kv,_t.size(),sizeof(Pair),[](const void* a,const void* b) {
         const Pair* ap = reinterpret_cast<const Pair*>(a);
         const Pair* bp = reinterpret_cast<const Pair*>(b);
         int d = ap->_v - bp->_v;
         return d ? d : (ap->_k - bp->_k);
      });
      for(int k=0;k < (int)_t.size();k++) {
         if (!_z->contains(_kv[k]._v))
            _y->remove(_kv[k]._k);
         else {
            if (_from == -1)
               _from = k;
            _to = k;
         }
      }
      if (_y->isBound())
         _z->assign(_t[_y->min()]);
      else {
         _y->propagateOnDomainChange(this);
         _z->propagateOnBoundChange(this);
      }      
   }
}

void Element1DBasic::propagate() 
{
   if (_y->isBound())
      _z->assign(_t[_y->min()]);
   else {
      int k = _from;
      while (k < (int)_t.size() && !_y->contains(_kv[k]._k)) ++k;
      if (k < (int)_t.size()) {
         _z->removeBelow(_kv[k]._v);
         _from = k;
      } else failNow();
      k = _to;
      while (k>=0 && !_y->contains(_kv[k]._k)) --k;
      if (k >= 0) {
         _z->removeAbove(_kv[k]._v);
         _to = k;
      } else failNow();
      k = _from;
      while (k < (int)_t.size() && _kv[k]._v < _z->min())
         _y->remove(_kv[k++]._k);
      _from = k;
      k = _to;
      while (k>=0 && _kv[k]._v > _z->max())
         _y->remove(_kv[k--]._k);
      _to = k;
   }
}

void Element1DBasic::print(std::ostream& os) const 
{
   os << "element1DBasic(" << _t << ',' << _y << ',' << _z << ')' << std::endl;
}

Element1DDC::Element1DDC(const std::vector<int>& array,var<int>::Ptr y,var<int>::Ptr z)
        : Constraint(y->getSolver()),_t(array),_y(y),_z(z)
{ setPriority(CNORMAL);}

int Element1DDC::findIndex(int target) const
{
    int l = 0,u = _nbv-1;
    while (l <= u) {
        int m = (l+u)/2;
        if (_values[m]._v == target)
            return m;
        else if (_values[m]._v > target)
            u = m - 1;
        else l = m + 1;
    }
    return _endOfList;
}

struct NoNotify : IntNotifier {
    void empty() {}
    void bind()  {}
    void change() {}
    void changeMin() {}
    void changeMax() {}
};

void Element1DDC::post()
{
    auto cps = _z->getSolver();
    _zOld = new (cps) BitDomain(cps->getStateManager(),cps->getStore(),_z->min(),_z->max());
    _yOld = new (cps) BitDomain(cps->getStateManager(),cps->getStore(),_y->min(),_y->max());
    _y->updateBounds(0,_t.size()-1);
    int yMin = _y->min(),yMax = _y->max();
    _endOfList = _y->min() - 1;
    std::vector<int> sorted;
    for(int k = yMin; k <= yMax;k++)
        if (_y->contains(k))
            sorted.push_back(_t[k]);
    std::sort(sorted.begin(),sorted.end());
    _nbv = 0;
    _values = new (cps) DCIndex[sorted.size()];
    int previous = sorted[0]-1; // so that the test below is false on first iteration
    for(auto v : sorted) {
        if (v!=previous) {
            _values[_nbv]._v = v;
            _values[_nbv]._k = _endOfList; // end-of-list marker. _list is built after
            _values[_nbv]._s = trail<int>(cps->getStateManager(),1);
            _nbv++;
        } else _values[_nbv-1]._s++; // one more index reaches that same previous value.
        previous = v;
    }
    int tMin = std::max(sorted[0],_z->min());         // use z.min and smallest in sorted
    int tMax = std::min(*(sorted.end()-1),_z->max()); // use z.max and largest in sorted
    _z->updateBounds(tMin,tMax);
    int last = 0;
    // prune _z based on the values that do appear in _values (i.e., in _t)
    for(int zk = tMin;zk <= tMax;zk++) {
        while (_values[last]._v < zk) last++; // skip until we are >= zk
        if (_values[last]._v > zk) // then zk is not in _values, delete it from _z. otherwise do nothing.
            _z->remove(zk);
    }
    // build all the linked lists in _list (header in _values._k set at EOL marker at start)
    _list = new (cps) int[yMax - yMin + 1];
    for(int yk=_y->min();yk <= _y->max();yk++)
        if (_y->contains(yk)) {
            int idx = findIndex(_t[yk]);  // locate the list that carries the value reachable by yk in _t
            _list[yk] = _values[idx]._k;  // set the list entry to that list
            _values[idx]._k = yk;         // and insert yk in front of that list (supports of _v)
        }
    // prune _y based on the values NOT in D(_z) that have supports.
    for(int k=0;k < _nbv;k++)
        if (!_z->contains(_values[k]._v)) {
            int link = _values[k]._k;
            while(link != _endOfList) {
                _y->remove(link);
                link = _list[link];
            }
        }
    // setup a copy of the domains of z,y to be able to identify which values were lost.
    // This is a quick reuse of BitDomain, which is why a Notifier is needed
    NoNotify nn;
    _zOld->removeBelow(_z->min(),nn);
    _zOld->removeAbove(_z->max(),nn);
    for(int k = _z->min(); k <= _z->max();k++)
        if (!_z->contains(k))
            _zOld->remove(k,nn);

    _yOld->removeBelow(_y->min(),nn);
    _yOld->removeAbove(_y->max(),nn);
    for(int k = _y->min(); k <= _y->max();k++)
        if (!_y->contains(k))
            _yOld->remove(k,nn);

    // Hook up listeners
    _y->propagateOnDomainChange(this);
    _z->propagateOnDomainChange(this);
}

void Element1DDC::zLostValue(int v)
{
    int k = findIndex(v);
    int link = _values[k]._k;
    while (link != _endOfList) {
        _y->remove(link);
        link = _list[link];
    }
}

void Element1DDC::yLostValue(int v)
{
    int k = findIndex(_t[v]);
    _values[k]._s -= 1;
    if (_values[k]._s == 0)
        _z->remove(_values[k]._v);
}

void Element1DDC::propagate()
{
    NoNotify nn;
    for(int i=_zOld->min(); i <= _zOld->max();i++) {
        if (_zOld->member(i) && !_z->contains(i)) { // i was lost from D(z) but we didn't know that -> value Lost Event
            zLostValue(i);
            _zOld->remove(i,nn);
        }
    }
    for(int i=_yOld->min();i <= _yOld->max();i++) {
        if (_yOld->member(i) && !_y->contains(i)) {
            yLostValue(i);
            _yOld->remove(i,nn);
        }
    }
}

void Element1DDC::print(std::ostream& os) const
{
    os << "element1DDC(" << _t << ',' << _y << ',' << _z << ')' << std::endl;
}

Element1D::Element1D(const std::vector<int>& array,var<int>::Ptr y,var<int>::Ptr z)
   : Constraint(y->getSolver()),_t(array),_y(y),_z(z)
{}

void Element1D::post()
{
   Matrix<int,2> t2({1,(int)_t.size()});
   for(auto j=0u;j< _t.size();j++)
      t2[0][j] = _t[j];
   auto x = Factory::makeIntVar(_y->getSolver(),0,0);
   auto c = new (_y->getSolver()) Element2D(t2,x,_y,_z);
   _y->getSolver()->post(c,false);
}



void Element1DVar::post()
{
   _y->updateBounds(1,(int)_array.size());
   for(size_t i = 1 ; i < _array.size(); i += 1)
   {
       _array[i]->propagateOnBoundChange(this);
   }
   _y->propagateOnDomainChange(this);
   _z->propagateOnBoundChange(this);
   propagate();
}

void Element1DVar::propagate()
{
   _zMin = _z->min();
   _zMax = _z->max();
   if (_y->isBound()) equalityPropagate();
   else {
      filterY();
      if (_y->isBound())
         equalityPropagate();
      else
         _z->updateBounds(_supMin->min(),_supMax->max());
   }
}

void Element1DVar::equalityPropagate()
{
   auto tVar = _array[_y->min()];
   tVar->updateBounds(_zMin,_zMax);
   _z->updateBounds(tVar->min(),tVar->max());
}

void Element1DVar::filterY()
{
   int min = INT32_MAX,max = INT32_MIN;
   int i = 0;
   for (int k=_y->min();k <= _y->max();k++)
      if (_y->contains(k))
         _yValues[i++] = k;
   while (i > 0) {
      int id = _yValues[--i];
      auto tVar =  _array[id];
      int tMin = tVar->min(),tMax = tVar->max();
      if (tMax < _zMin || tMin > _zMax)
         _y->remove(id);
      else {
         if (tMin < min) {
            min = tMin;
            _supMin = tVar;
         }
         if (tMax > max) {
            max = tMax;
            _supMax = tVar;
         }
      }
   }
}