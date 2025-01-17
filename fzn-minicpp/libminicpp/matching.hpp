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

#ifndef __MATCHING_H
#define __MATCHING_H

#include <algorithm>
#include <stack>
#include <iomanip>
#include <stdint.h>
#include "intvar.hpp"

class PGraph {
        struct VTag {
            int disc;
            int low;
            bool held;
        };
        const int _nbVar,_nbVal,_V,_sink;
        int _minVal,_maxVal;
        int* _match;
        int* _varFor;
        Factory::Veci::pointer _x;
        template <typename B>
        void SCCFromVertex(const int v,B body,int& time,int u,VTag tags[],int st[],int& top);
        template <typename B>
        void SCCUtil(B body,int& time,int u,VTag tags[],int st[],int& top);
    public:
        PGraph(int nbVar,int minVal,int maxVal,int match[],int varFor[],Factory::Veci::pointer x)
                : _nbVar(nbVar),
                  _nbVal(maxVal - minVal + 1),
                  _V(nbVar + _nbVal + 1),
                  _sink(nbVar + _nbVal),
                  _minVal(minVal),
                  _maxVal(maxVal),
                  _match(match),
                  _varFor(varFor),
                  _x(x) {}
        void setLiveValues(int min,int max) { _minVal = min;_maxVal = max;}
        template <typename B> void SCC(B body); // apply body to each SCC
};

class MaximumMatching {
        Storage::Ptr _store;
        Factory::Veci& _x;
        int* _match,*_varSeen;
        int _min,_max;
        int _valSize;
        int*  _valMatch,*_valSeen;
        int _szMatching;
        int _magic;
        void findInitialMatching();
        int findMaximalMatching();
        bool findAlternatingPathFromVar(int i);
        bool findAlternatingPathFromVal(int v);
    public:
        MaximumMatching(Factory::Veci& x,Storage::Ptr store)
                : _store(store),_x(x) {}
        ~MaximumMatching();
        void setup();
        int compute(int result[]);
};

template <typename B>
void PGraph::SCCFromVertex(const int v,B body,int& time,int u, VTag tags[],
                           int st[],int& top)
{
    if (tags[v].disc == -1)  {
        SCCUtil(body,time,v,tags,st,top);
        tags[u].low = std::min(tags[u].low, tags[v].low);
    }
    else if (tags[v].held)
        tags[u].low = std::min(tags[u].low, tags[v].disc);
}


template <typename B>
void PGraph::SCCUtil(B body,int& time,int u, VTag tags[],int st[],int& top)
{
    ++time;
    tags[u] = {time,time,true};
    st[top++] = u;
    if (u < _nbVar) { // u is a variable [0.._nbVar)
        const int xuMin = _x[u]->min(),xuMax = _x[u]->max(),mu = _match[u];
        for(int k = xuMin ; k <= xuMax ; ++k) {
            if (mu != k && _x[u]->containsBase(k)) {
                const int v = k - _minVal + _nbVar;
                SCCFromVertex(v,body,time,u,tags,st,top); // v is a value node in the pseudo-graph
            }
        }
    } else if (u < _nbVar + _nbVal) { // u is a value. Exactly one edge out (to sink or var when matched)
        const int v = (_varFor[u - _nbVar] == -1) ? _sink : _varFor[u - _nbVar];
        SCCFromVertex(v,body,time,u,tags,st,top);
    } else { // u == sink: edge set going to matched values
        const int lastShiftedVal = _maxVal - _minVal;
        for(int k = 0; k <= lastShiftedVal;++k) {
            if (_varFor[k] != -1) {
                const int v = _nbVar + k;  // this is the *value* node
                SCCFromVertex(v,body,time,u,tags,st,top);
            }
        }
    }
    if (tags[u].low == tags[u].disc)  {
        int oldTop = top;
        while (st[top-1] != u)
            tags[st[--top]].held = false;
        tags[st[--top]].held = false;
        body(oldTop - top,st+top);
    }
}

template <typename B> void PGraph::SCC(B body)
{
    VTag tags[_V];
    int st[_V];
    int top = 0,time = 0;
    for (int i = 0; i < _V; i++)
        tags[i] = {-1,-1,false};

    for (int i = 0; i < _V; i++)
        if (tags[i].disc == -1)
            SCCUtil(body,time,i,tags,st,top);
}

#endif
