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

#ifndef __ACSTR_H
#define __ACSTR_H

#include <algorithm>
#include "handle.hpp"
#include "trailable.hpp"

class CPSolver;

class Constraint {

    public:
        enum AsyncState : unsigned char {ToOffload, ToRetrieve, Done};

    private:
       bool         _scheduled;
       bool      _asynchronous;
       unsigned char     _prio;
       trail<bool>     _active;
       AsyncState  _asyncState;

    public:
       static const unsigned char CHIGH = 3;
       static const unsigned char CNORMAL= 2;
       static const unsigned char CLOW  = 1;
       static const unsigned char ASYNC = 0;
       typedef handle_ptr<Constraint> Ptr;
       Constraint(handle_ptr<CPSolver> cp);
       virtual ~Constraint() {}
       virtual void post() = 0;
       virtual void propagate() {}
       virtual void print(std::ostream& os) const {}
       void setPriority(unsigned char p) { _prio = p;}
       unsigned char getPriority() const { return _prio;}
       void setScheduled(bool s)    { _scheduled = s;}
       bool isScheduled() const     { return _scheduled;}
       void setAsynchronous(bool a) { _asynchronous = a;}
       bool isAsynchronous() const  { return _asynchronous;}
       void setActive(bool a)       { _active = a;}
       bool isActive() const        { return _active;}
       void setAsyncState(AsyncState a)       { _asyncState = a;}
       AsyncState getAsyncState() const       {return _asyncState;}
       virtual void offload() {};
       virtual void retrieve() {};
};

class Objective
{
private:
    std::vector<std::function<void(void)>> _failureListeners;
public:
    typedef handle_ptr<Objective> Ptr;
    virtual void tighten() = 0;
    virtual int value() const = 0;
    template <class F> void onFailure(F f)  {_failureListeners.emplace_back(std::move(f));}
    void notifyFailure()  {for_each(_failureListeners.begin(),_failureListeners.end(),[](std::function<void(void)>& f) { f();});}
};

#endif
