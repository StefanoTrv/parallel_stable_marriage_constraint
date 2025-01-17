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

#ifndef __STATE_H
#define __STATE_H

#include <functional>
#include "handle.hpp"

class StateManager {
public:
   StateManager() {}
   virtual ~StateManager() {}
   typedef handle_ptr<StateManager> Ptr;
   virtual void enable() {}
   virtual void saveState() = 0;
   virtual void clear() = 0;
   virtual void restoreState() = 0;
   virtual void withNewState(const std::function<void(void)>& body) = 0;
};

#endif
