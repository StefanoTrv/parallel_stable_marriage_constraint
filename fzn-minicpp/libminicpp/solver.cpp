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

#include "solver.hpp"
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <typeindex>

CPSolver::CPSolver()
    : _sm(new Trailer),
      _store(new Storage(_sm))
{
    _varId  = 0;
    _propagations = 0;
}

CPSolver::~CPSolver()
{
   _iVars.clear();
   _store.dealloc();
   _sm.dealloc();
   std::cout << "CPSolver::~CPSolver(" << this << ")" << std::endl;
}

void CPSolver::post(Constraint::Ptr c, bool enforceFixPoint)
{
    if (!c)
        return;
    c->post();
    if (enforceFixPoint)
        fixpoint();
}

void CPSolver::registerVar(AVar::Ptr avar)
{
   avar->setId(_varId++);
   _iVars.push_back(avar);
}

void CPSolver::notifyFixpoint()
{
   for(auto& body : _onFix)
      body();
}

void CPSolver::fixpoint()
{
    TRYFAIL
        notifyFixpoint();
        while (!_queue.empty())
        {
            auto c = _queue.deQueue();

            if(not c->isAsynchronous())
            {
                c->setScheduled(false);
                if (c->isActive())
                {
                    c->propagate();
                    _propagations += 1;
                }
            }
            else
            {
                if (c->isActive())
                {
                    switch (c->getAsyncState())
                    {
                        case Constraint::AsyncState::ToOffload:
                            c->setAsyncState(Constraint::AsyncState::ToRetrieve);
                            _queue.enQueue(c);
                            c->offload();
                            _propagations += 1;
                            break;
                        case Constraint::AsyncState::ToRetrieve:
                            c->setScheduled(false);
                            c->setAsyncState(Constraint::AsyncState::Done);
                            c->retrieve();
                            break;
                        default:
                            throw std::runtime_error("Unexpected asynchronous state");
                    }
                }
                else
                {
                    c->setScheduled(false);
                }
            }

        }
    ONFAIL
        while (not _queue.empty())
        {
            _queue.deQueue()->setScheduled(false);
        }
        assert(_queue.size() == 0);
        failNow();
    ENDFAIL
}
