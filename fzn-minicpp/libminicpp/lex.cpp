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


#include "lex.hpp"
#include "constraint.hpp"
#include <algorithm>

#define LEX_XEQ_GEQY(i) (_x[(i)]->min() == _y[(i)]->max())
#define LEX_XEQ_LEQY(i) (_x[(i)]->max() == _y[(i)]->min())
#define LEX_XLY(i)      (_x[(i)]->max() < _y[(i)]->min())
#define LEX_XEQY(i)     (_x[(i)]->isBound() && _y[(i)]->isBound() && _x[(i)]->min() == _y[(i)]->min())
#define LEX_XGY(i)      (_x[(i)]->min() > _y[(i)]->max())
#define LEX_XLEQY(i)    (_x[(i)]->max() == _y[(i)]->min() && _x[(i)]->min() < _y[(i)]->max())
#define LEX_XGEQY(i)    (_y[(i)]->max() == _x[(i)]->min() && _y[(i)]->min() < _x[(i)]->max())

void LexLeq::post() 
{
   using namespace Factory;
   auto cp = _x[0]->getSolver();
   int up = _sz - 1,i = 0;
   while (i <= up && LEX_XEQ_GEQY(i)) {
      int xi = _x[i]->min();
      _x[i]->assign(xi);
      _y[i]->assign(xi);
      ++i;
   }
   _q = i;
   if (i > up || LEX_XLY(i))
      return;
   _x[_q]->removeAbove(_y[_q]->max());
   _y[_q]->removeBelow(_x[_q]->min());
   _r = ++i;
   while (i <= up && LEX_XEQY(i))
      ++i;
   _r = i;
   if (i > up || LEX_XLY(i)) {
      cp->post(_x[_q] <= _y[_q]);
      return;
   } else if (LEX_XGY(i)) {
      cp->post(_x[_q] < _y[_q]);
      return;
   } else if (LEX_XLEQY(i)) {
      _s = ++i;
      while(i <= up && LEX_XEQ_LEQY(i))
         ++i;
      _s = i;
      if (i > up || LEX_XLY(i)) {
         cp->post(_x[_q] <= _y[_q]);
         return ;
      }
      listenFrom(_q);
      _u = 3;
   } else if (LEX_XGEQY(i)) {
      _s = ++i;
      while (i <= up && LEX_XEQ_GEQY(i))
         ++i;
      _s = i;
      if (i <= up && LEX_XGY(i)) {
         cp->post(_x[_q] < _y[_q]);
         return ;
      }
      listenFrom(_q);
      _u = 4;
   } else {
      listenFrom(_q);
      _u = 2;
   }
}

void LexLeq::listenFrom(int ofs)
{
   for(int k=ofs;k < _sz;k++) {
      if (!_x[k]->isBound()) 
         _x[k]->whenBoundsChange([this,k](){
            propagateFrom(k);
         });
      if (!_y[k]->isBound()) 
         _y[k]->whenBoundsChange([this,k]() {
            propagateFrom(k);
         });
   } 
}

void LexLeq::propagateFrom(int k)
{
   using namespace Factory;
   if (!isActive()) return;
   auto cp = _x[0]->getSolver();
   int up = _sz - 1,i = k;
   if (k == _q) goto STATE1;
   else if (k == _r) goto STATE2;
   else if (_u == 3 && (k == _s || (k < _s && _x[k]->max() != _y[k]->min()))) goto STATE3;
   else if (_u == 4 && (k == _s || (k < _s && _x[k]->min() != _y[k]->max()))) goto STATE4;
   else return;
 STATE1:
   while (i <= up && LEX_XEQ_GEQY(i)) {
      int xi = _x[i]->min();
      _x[i]->assign(xi);
      _y[i]->assign(xi);
      ++i;
   }
   _q = i;
   if (i > up || LEX_XLY(i)) {
      setActive(false);
      return;
   }
   _x[_q]->removeAbove(_y[_q]->max());
   _y[_q]->removeBelow(_x[_q]->min());
   _r = i = std::max(i+1,_r.value());
 STATE2:
   while (i <= up && LEX_XEQY(i))
      ++i;
   _r = i;
   if (i > up || LEX_XLY(i)) {
      setActive(false);
      if (_y[_q]->isBound())
         _x[_q]->removeAbove(_y[_q]->min());
      else
         cp->post(_x[_q] <= _y[_q]);
      return;
   } else if (LEX_XGY(i)) {
      setActive(false);
      if (_y[_q]->isBound())
         _x[_q]->removeAbove(_y[_q]->min() - 1);
      else
         cp->post(_x[_q] < _y[_q]);
      return;
   } else if (LEX_XLEQY(i)) {
      _s = i = std::max(i+1,_s.value());
      goto STATE3;
   } else if (LEX_XGEQY(i)) {
      _s = i = std::max(i+1,_s.value());
      goto STATE4;
   }
   _u = 2;
   return;
 STATE3:
   while (i <= up && LEX_XEQ_LEQY(i))
      ++i;
   _s = i;
   if (i > up || LEX_XLY(i)) {
      setActive(false);
      if (_y[_q]->isBound())
         _x[_q]->removeAbove(_y[_q]->min());
      else
         cp->post(_x[_q] <= _y[_q]);
      return;
   }
   _u = 3;
   return;
 STATE4:
   while (i <= up && LEX_XEQ_GEQY(i))
      ++i;
   _s = i;
   if (i <= up && LEX_XGY(i)) {
      setActive(false);
      if (_y[_q]->isBound())
         _x[_q]->removeAbove(_y[_q]->min() - 1);
      else
         cp->post(_x[_q] < _y[_q]);
      return;
   }
   _u = 4;
}
