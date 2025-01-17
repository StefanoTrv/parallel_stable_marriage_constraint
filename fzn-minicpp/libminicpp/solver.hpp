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

#ifndef __SOLVER_H
#define __SOLVER_H

#include <list>
#include <deque>
#include <functional>
#include <stdlib.h>
#include <setjmp.h>

#include "handle.hpp"
#include "fail.hpp"
#include "store.hpp"
#include "avar.hpp"
#include "acstr.hpp"
#include "trailable.hpp"

typedef std::reference_wrapper<std::function<void(void)>> Closure;
class Controller;

class DEPQueue {
   std::deque<Constraint::Ptr>  _q[4];
public:
   DEPQueue() {}

   void enQueue(Constraint::Ptr& c)
   {
       if (not c->isAsynchronous())
       {
           _q[c->getPriority()].emplace_back(c);
       }
       else
       {
           _q[Constraint::ASYNC].emplace_back(c);
       }
   }

   bool empty() const
   {
        return _q[Constraint::CHIGH].empty() and
               _q[Constraint::CNORMAL].empty() and
               _q[Constraint::CLOW].empty() and
               _q[Constraint::ASYNC].empty();
   }
   auto size() const
   {
        return _q[Constraint::CHIGH].size() +
               _q[Constraint::CNORMAL].size() +
               _q[Constraint::CLOW].size() +
               _q[Constraint::ASYNC].size();
   }
   Constraint::Ptr deQueue()
   {
        if (not _q[Constraint::CHIGH].empty())
        {
            auto c = _q[Constraint::CHIGH].front();
            _q[Constraint::CHIGH].pop_front();
            return c;
        }
        else if (not _q[Constraint::CNORMAL].empty())
        {
            auto c = _q[Constraint::CNORMAL].front();
            _q[Constraint::CNORMAL].pop_front();
            return c;
        }
        else if (not _q[Constraint::CLOW].empty())
        {
            auto c = _q[Constraint::CLOW].front();
            _q[Constraint::CLOW].pop_front();
            return c;
        }
        else
        {
            auto c = _q[Constraint::ASYNC].front();
            _q[Constraint::ASYNC].pop_front();
            return c;
        }
   }
};

class CPSolver {
    Trailer::Ptr                  _sm;
    Storage::Ptr               _store;
    std::list<AVar::Ptr>       _iVars;
    DEPQueue                   _queue;
    std::list<std::function<void(void)>>  _onFix;
    long                  _afterClose;
    int                        _varId;
    unsigned long long  _propagations;
public:
    template<typename T> friend class var;
    typedef handle_ptr<CPSolver> Ptr;
    CPSolver();
    ~CPSolver();
    Trailer::Ptr getStateManager()       { return _sm;}
    Storage::Ptr getStore()              { return _store;}
    unsigned long long getPropagations() {return _propagations;};
    void registerVar(AVar::Ptr avar);
    void schedule(Constraint::Ptr& c)
    {
        if(not c->isAsynchronous())
        {
            if (c->isActive() && !c->isScheduled())
            {
                c->setScheduled(true);
                _queue.enQueue(c);
            }
        }
        else
        {
            if (c->isActive())
            {
                c->setAsyncState(Constraint::AsyncState::ToOffload);
                if(not c->isScheduled())
                {
                    c->setScheduled(true);
                    _queue.enQueue(c);
                }

            }
        }
    }
    void onFixpoint(std::function<void(void)>& cb) { _onFix.emplace_back(cb);}
    void notifyFixpoint();
   //void tighten();
    void fixpoint();
   //void fixpointNT();
    void post(Constraint::Ptr c,bool enforceFixPoint=true);
   //void fail();
    friend void* operator new(std::size_t sz,CPSolver::Ptr e);
    friend void* operator new[](std::size_t sz,CPSolver::Ptr e);
    friend std::ostream& operator<<(std::ostream& os,const CPSolver& s) {
       return os << "CPSolver(" << &s << ")" << std::endl;
    }
};

namespace Factory {
   inline CPSolver::Ptr makeSolver() { return new CPSolver;}
};

inline void* operator new(std::size_t sz,CPSolver::Ptr e)
{
   return e->_store->allocate(sz);
}

inline void* operator new[](std::size_t sz,CPSolver::Ptr e)
{
   return e->_store->allocate(sz);
}


#endif
