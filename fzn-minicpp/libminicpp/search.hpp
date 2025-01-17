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

#ifndef __SEARCH_H
#define __SEARCH_H

#define TRACE(X)

#include <vector>
#include <initializer_list>
#include <functional>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "solver.hpp"
#include "constraint.hpp"
#include "RuntimeMonitor.hpp"

class Branches {
   std::vector<std::function<void(void)>> _alts;
public:
   Branches(const Branches& b) : _alts(std::move(b._alts)) {}
   Branches(std::vector<std::function<void(void)>> alts) : _alts(alts) {}
   Branches(std::initializer_list<std::function<void(void)>> alts) { _alts.insert(_alts.begin(),alts.begin(),alts.end());}
   Branches(std::function<void(void)> f) {_alts.push_back(std::move(f));}
   ~Branches() {}
   std::vector<std::function<void(void)>>::iterator begin() { return _alts.begin();}
   std::vector<std::function<void(void)>>::iterator end()   { return _alts.end();}
   size_t size() const { return _alts.size();}
   void add(std::function<void(void)> f) {_alts.push_back(std::move(f));};
};

/*
class Chooser {
   std::function<Branches(void)> _sel;
public:
   Chooser(std::function<Branches(void)> sel) : _sel(sel) {}
   Branches operator()() { return _sel();}
};
*/

class SearchStatistics
{
    protected:
        unsigned long long nodes;
        unsigned long long failures;
        unsigned long long solutions;
        unsigned long long tighteningFail;
        RuntimeMonitor::HRClock startTime;
        RuntimeMonitor::HRClock searchStartTime;
        RuntimeMonitor::HRClock searchEndTime;
        bool completed;

    public:
       SearchStatistics() :
               nodes(0),
               solutions(0),
               failures(0),
               tighteningFail(0),
               completed(false)
       {}
       void incrFailures() noexcept {failures += 1;}
       void incrNodes() noexcept { nodes += 1;}
       void incrSolutions() noexcept {solutions += 1;}
       void incrTighteningFail() noexcept { tighteningFail += 1;}
       void setStartTime() noexcept {startTime = RuntimeMonitor::now();}
       void setSearchStartTime() noexcept { searchStartTime = RuntimeMonitor::now();}
       void setSearchEndTime() noexcept { searchEndTime = RuntimeMonitor::now();};
       void setCompleted() noexcept { completed = true;};
       bool getCompleted() noexcept { return completed;};
       unsigned long long getFailures() const noexcept {return failures;};
       unsigned long long getNodes() const noexcept {return nodes;};
       unsigned long long getSolutions() const noexcept {return solutions;};
       unsigned long long getTighteningFail() const noexcept {return tighteningFail;};
       double getInitTime() const noexcept {return RuntimeMonitor::elapsedSeconds(startTime, searchStartTime);};
       double getSolveTime() const noexcept {return RuntimeMonitor::elapsedSeconds(searchStartTime, searchEndTime);};
       double getRunningTime() const noexcept {return RuntimeMonitor::elapsedSeconds(startTime, RuntimeMonitor::now());};
       friend std::ostream& operator<<(std::ostream& os,const SearchStatistics& ss)
       {
            return os
                    << std::fixed << std::setprecision(3)
                    << "Solving time = " << ss.getSolveTime() << " s" << std::endl
                    << "Solutions = " << ss.getSolutions() << std::endl
                    << "Nodes = " << 1 + ss.getNodes() << std::endl
                    << "Failures = " << ss.getFailures() - ss.getTighteningFail() << std::endl;
       };
    };

typedef std::function<bool(const SearchStatistics&)> Limit;

class DFSearch {
    StateManager::Ptr                      _sm;
    std::function<Branches(void)>   _branching;
    std::vector<std::function<void(void)>>    _solutionListeners;
    std::vector<std::function<void(void)>>    _nodeListeners;
    std::vector<std::function<void(void)>>    _failureListeners;
    trail<int> _depth;
    int _peakDepth = 0;
    void dfs(SearchStatistics& stats,const Limit& limit);
public:
   DFSearch(CPSolver::Ptr cp,std::function<Branches(void)>&& b)
      : _sm(cp->getStateManager()),_branching(std::move(b)), _depth(cp->getStateManager(),0) {
      _sm->enable();
   }
   DFSearch(StateManager::Ptr sm,std::function<Branches(void)>&& b)
      : _sm(sm),_branching(std::move(b)) {
      _sm->enable();
   }
   template <class B> void onSolution(B c) { _solutionListeners.emplace_back(std::move(c));}
   template <class B> void onBranch(B c)  { _nodeListeners.emplace_back(std::move(c));}
   template <class B> void onFailure(B c)  { _failureListeners.emplace_back(std::move(c));}
   void notifySolution() { for_each(_solutionListeners.begin(),_solutionListeners.end(),[](std::function<void(void)>& c) { c();});}
   void notifyNode()  { for_each(_nodeListeners.begin(), _nodeListeners.end(), [](std::function<void(void)>& c) { c();});}
   void notifyFailure()  { for_each(_failureListeners.begin(),_failureListeners.end(),[](std::function<void(void)>& c) { c();});}
   int getPeakDepth()const  {return  _peakDepth;};
   SearchStatistics solve(SearchStatistics& stat,Limit limit);
   SearchStatistics solve(SearchStatistics& stat);
   SearchStatistics solve(Limit limit);
   SearchStatistics solve();
   SearchStatistics solveSubjectTo(Limit limit,std::function<void(void)> subjectTo);
   SearchStatistics optimize(Objective::Ptr obj,SearchStatistics& stat,Limit limit);
   SearchStatistics optimize(Objective::Ptr obj,SearchStatistics& stat);
   SearchStatistics optimize(Objective::Ptr obj,Limit limit);
   SearchStatistics optimize(Objective::Ptr obj);
   SearchStatistics optimizeSubjectTo(Objective::Ptr obj,Limit limit,std::function<void(void)> subjectTo);
};

template<class B>
std::function<Branches(void)> land(std::initializer_list<B> allB) {
   std::vector<B> vec(allB);
   return land(vec);
}

template<class B>
std::function<Branches(void)> land(std::vector<B> vec)
{
    return [vec]()
    {
        for(size_t i = 0; i < vec.size(); i += 1)
        {
            auto br = vec[i]();
            if (br.size() != 0)
                return br;
        }
        return Branches({});
    };
}

inline Branches operator|(std::function<void(void)> b0, std::function<void(void)> b1) {
    return Branches({ b0,b1 });
}

template<class Container,typename Predicate, typename Fun>
typename Container::value_type selectMin(Container& c,Predicate test, Fun f)
{
   auto from = c.begin();
   auto to = c.end();
   auto min = to;
   for(; from != to; from += 1)
   {
       if (test(*from))
       {
           auto fv = f(*from);
           if (min == to || fv < f(*min))
               min = from;
       }
   }
   if (min == to)
      return typename Container::value_type();
   else 
      return *min;
}

template <class Container> std::function<Branches(void)> firstFail(CPSolver::Ptr cp,Container& c) {
    using namespace Factory;
    return [=]() {
        auto sx = selectMin(c,
                            [](const auto& x) { return x->size() > 1;},
                            [](const auto& x) { return x->size();});
        if (sx) {
            int v = sx->min();
            return [cp,sx,v] { return cp->post(sx == v);}
                   |  [cp,sx,v] { return cp->post(sx != v);};
        } else return Branches({});
    };
}

template<class Container,typename Predicate, typename Fun>
typename Container::value_type selectMax(Container& c,Predicate test, Fun f)
{
    auto from = c.begin();
    auto to = c.end();
    auto max = to;
    for(; from != to; from += 1)
    {
        if (test(*from))
        {
            auto fv = f(*from);
            if (max == to || fv > f(*max))
                max = from;
        }
    }
    if (max == to)
        return typename Container::value_type();
    else
        return *max;
}

template<class Vars, class Var>
Var first_fail(Vars const & vars)
{
    return selectMin(
                vars,
                [](const auto& x) { return x->size() > 1;},
                [](const auto& x) { return x->size();}
            );
}

template<class Vars, class Var>
Var input_order(Vars const & vars)
{
    return selectMin(
            vars,
            [](const auto& x) { return x->size() > 1;},
            [&](const auto& x)
            {
                auto it = std::find(vars.begin(), vars.end(), x);
                return std::distance(vars.begin(), it);
            }
    );
}

template<class Vars, class Var>
Var smallest(Vars const & vars)
{
    return selectMin(
            vars,
            [](const auto& x) { return x->size() > 1;},
            [](const auto& x) { return x->min();}
    );
}

template<class Vars, class Var>
Var largest(Vars const & vars)
{
    return selectMax(
            vars,
            [](const auto& x) { return x->size() > 1;},
            [](const auto& x) { return x->max();}
    );
}


// Values selections
template<class Var>
Branches indomain_min(CPSolver::Ptr cp, Var var)
{
    using namespace Factory;

    if (var)
    {
        auto val = var->min();
        return [cp,var,val] { TRACE(std::cout << "%% Choosing x" << var->getId() << " == "<< val << std::endl;) return cp->post(var == val);} |
               [cp,var,val] { TRACE(std::cout << "%% Choosing x" << var->getId() << " != "<< val << std::endl;) return cp->post(var != val);};
    }
    else
    {
        return Branches({});
    }
}

template <class Var>
Branches indomain_max(CPSolver::Ptr cp, Var var)
{
    using namespace Factory;

    if (var)
    {
        auto val = var->max();
        return [cp,var,val] {return cp->post(var == val);} |
               [cp,var,val] {return cp->post(var != val);};
    }
    else
    {
        return Branches({});
    }
}

template <class Var>
Branches random(CPSolver::Ptr cp, Var var)
{
    using namespace Factory;

    if (var)
    {
        auto b = Branches({});
        for (auto i  = 0; i < 3; i += 1)
        {
            b.add([cp,var] {
                auto var_size = var->size();
                int val_pos = rand() % var->size() + 1;
                auto val = var->getIthVal(val_pos);
                return cp->post(var == val);}
            );
        };
        return b;
    }
    else
    {
        return Branches({});
    }
}

template <class Var>
Branches lin_decr(CPSolver::Ptr cp, Var var)
{
    using namespace Factory;

    if (var)
    {
        auto b = Branches({});
        for (auto i  = 0; i < 3; i += 1)
        {
            b.add([cp,var] {
                float uniform_random01 = ((float) rand()) / (float) RAND_MAX;
                float lin_dec_sample = sqrt(1 - uniform_random01);
                int val_pos = round(lin_dec_sample * var->size()) + 1;
                auto val = var->getIthVal(val_pos);
                return cp->post(var == val);}
            );
        };
        return b;
    }
    else
    {
        return Branches({});
    }
}

template <class Var>
Branches indomain_split(CPSolver::Ptr cp, Var var)
{
    using namespace Factory;

    if (var)
    {
        auto val = (var->max() + var->min()) / 2;
        return [cp,var,val] {return cp->post(var <= val);} |
               [cp,var,val] {return cp->post(var >= val + 1);};
    }
    else
    {
        return Branches({});
    }
}


#endif
