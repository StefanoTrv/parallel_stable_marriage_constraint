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

#ifndef __AVAR_H
#define __AVAR_H

#include <memory>
#include "handle.hpp"

class AVar {
protected:
    virtual void setId(int id) = 0;
    friend class CPSolver;
public:
    typedef handle_ptr<AVar> Ptr;
};


template<typename T> class var {};

#endif
