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

#ifndef __REVLIST_H
#define __REVLIST_H

#include "trailable.hpp"
#include "store.hpp"

template<class T> class trailList {
    Trailer::Ptr _sm;
    Storage::Ptr _store;
public:
    struct revNode {
        trailList<T>*   _owner;
        trail<revNode*> _prev;
        trail<revNode*> _next;
        T            _value;
        revNode(trailList<T>* own,Trailer::Ptr ctx,revNode* p,revNode* n,T&& v)
            : _owner(own),_prev(ctx,p),_next(ctx,n),_value(std::move(v)) {
            if (p) p->_next = this;
            if (n) n->_prev = this;
        }
        void detach() {
            revNode* p = _prev;
            revNode* n = _next;
            if (p) p->_next = n;
            else _owner->_head = n;
            if (n) n->_prev = p;
        }
    };
    class iterator {
        friend class trailList<T>;
        revNode* _cur;
    protected:
        iterator(revNode* c) : _cur(c) {}
    public:
        T operator*() const  { return _cur->_value;}
        T& operator*()       { return _cur->_value;}
        const iterator& operator++() { _cur = _cur->_next;return *this;}
        iterator operator++(int) { iterator copy(_cur);_cur = _cur->_next;return copy;}
        bool operator==(const iterator& other) const { return _cur == other._cur;}
        bool operator!=(const iterator& other) const { return _cur != other._cur;}
    };
private:
    trail<revNode*> _head;
public:
    trailList(Trailer::Ptr sm,Storage::Ptr store) : _sm(sm),_store(store),_head(sm,nullptr) {}
    ~trailList() {
        _head = nullptr;
    }
    revNode* emplace_back(T&& v) {
        // Allocate list node on the stack allocator
        return _head = new (_store) revNode(this,_sm,nullptr,_head,std::move(v));
    }
    iterator begin()  { return iterator(_head);}
    iterator end()    { return iterator(nullptr);}
    friend std::ostream& operator<<(std::ostream& os,const trailList<T>& rl) {
        revNode* cur = rl._head;
        while (cur) {
            os << "," << cur;
            cur = cur->_next;
        }
        return os;
    }
};

typedef trailList<Constraint::Ptr>::revNode TLCNode;

#endif
