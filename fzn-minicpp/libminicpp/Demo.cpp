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

#include <iostream>
#include "solver.hpp"
#include "intvar.hpp"
#include "constraint.hpp"
#include "search.hpp"

int main(int argc,char* argv[])
{
    using namespace std;
    using namespace Factory;

    // Arguments check
    if (argc != 2)
    {
        cout << "Usage example: minicpp-demo 8" << endl;
        exit(EXIT_FAILURE);
    }
    int const n = atoi(argv[1]);
    if (not n)
    {
        cout << "ERROR: Expected integer argument" << endl;
        exit(EXIT_FAILURE);
    }

    // Problem definition
    CPSolver::Ptr cp  = Factory::makeSolver();
    auto q = Factory::intVarArray(cp,n,1,n); // Rows
    auto ud = Factory::intVarArray(cp,n,[q](int i){return q[i] + i;}); // Upwards diagonals
    auto dd = Factory::intVarArray(cp,n,[q](int i){return q[i] - i;}); // Downwards diagonals
    cp->post(Factory::allDifferentAC(q));
    cp->post(Factory::allDifferentAC(ud));
    cp->post(Factory::allDifferentAC(dd));

    // Search strategy
    DFSearch search(cp,[=]() {
        // Variable selection
        auto x = selectMin(q, [](const auto& x) { return x->size() > 1;}, [](const auto& x) { return x->size();});
        // Value selection
        if (x)
        {
            int c = x->min();
            return  [=] { cp->post(x == c);} | [=] { cp->post(x != c);};
        }
        else
        {
            return Branches({});
        }
    });

    // Initialize statistics
    SearchStatistics stats;
    search.onBranch([&](){ stats.incrNodes();});
    search.onFailure([&](){stats.incrFailures();});
    search.onSolution([&](){stats.incrSolutions();});

    // Output
    search.onSolution([&q](){cout << "Solution = " << q << endl;});

    // Search
    stats.setSearchStartTime();
    search.solve(stats);
    stats.setSearchEndTime();

    // Print statistics
    cout << endl;
    cout << stats;

    return EXIT_SUCCESS;
}
