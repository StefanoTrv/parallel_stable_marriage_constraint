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

#include "search.hpp"
#include "intvar.hpp"
#include "constraint.hpp"
#include <string>
#include <inttypes.h>
#include <vector>

class StopException {};

typedef std::function<void(void)> VVFun;

SearchStatistics DFSearch::solve(SearchStatistics& stats,Limit limit)
{
    _sm->withNewState(VVFun([this,&stats,&limit]() {
                               try {
                                  dfs(stats,limit);
                                  stats.setCompleted();
                               } catch(StopException& sx) {}
                            }));
    return stats;
}

SearchStatistics DFSearch::solve(SearchStatistics& stats)
{
    return solve(stats,[](const SearchStatistics& ss) {return false;});
}

SearchStatistics DFSearch::solve(Limit limit)
{
    SearchStatistics stats;
    return solve(stats,limit);
}

SearchStatistics DFSearch::solve()
{
    SearchStatistics stats;
    return solve(stats,[](const SearchStatistics& ss) { return false;});
}

SearchStatistics DFSearch::solveSubjectTo(Limit limit,std::function<void(void)> subjectTo)
{
    SearchStatistics stats;
    _sm->withNewState(VVFun([this,&stats,&limit,&subjectTo]() {
                               try {
                                  subjectTo();
                                  dfs(stats,limit);
                                  stats.setCompleted();
                               } catch(StopException& sx) {}
                            }));
    return stats;
}

SearchStatistics DFSearch::optimize(Objective::Ptr obj,SearchStatistics& stats,Limit limit)
{
   onSolution([obj] {obj->tighten();});
   return solve(stats,limit);
}

SearchStatistics DFSearch::optimize(Objective::Ptr obj,SearchStatistics& stats)
{
   onSolution([obj] {obj->tighten();});
   return solve(stats,[](const SearchStatistics& ss) { return false;});
}

SearchStatistics DFSearch::optimize(Objective::Ptr obj,Limit limit)
{
   SearchStatistics stats;
   onSolution([obj] {obj->tighten();});
   return solve(stats,limit);
}

SearchStatistics DFSearch::optimize(Objective::Ptr obj)
{
   return optimize(obj,[](const SearchStatistics& ss) { return false;});
}

SearchStatistics DFSearch::optimizeSubjectTo(Objective::Ptr obj,Limit limit,std::function<void(void)> subjectTo)
{
   SearchStatistics stats;
   _sm->withNewState(VVFun([this,&stats,obj,&limit,&subjectTo]() {
                              try {
                                 subjectTo();
                                 stats = optimize(obj,limit);
                              } catch(StopException& sx) {}
                           }));
   return stats;
}

void DFSearch::dfs(SearchStatistics& stats,const Limit& limit)
{
    Branches branches = _branching();
    if (branches.size() == 0)
    {
        TRYFAIL
                notifySolution();
        ONFAIL
                notifyFailure();
        ENDFAIL
    }
    else
    {
        _depth += 1;
        _peakDepth = std::max(_depth.value(),_peakDepth);
        notifyNode();
        for (auto cur = branches.begin(); cur != branches.end() and !limit(stats); cur++)
        {
            _sm->saveState();
            TRYFAIL
                    (*cur)();
                    dfs(stats, limit);
            ONFAIL
                    notifyFailure();
            ENDFAIL
            _sm->restoreState();
        }
        if (limit(stats))
        {
            throw StopException();
        }
    }
}