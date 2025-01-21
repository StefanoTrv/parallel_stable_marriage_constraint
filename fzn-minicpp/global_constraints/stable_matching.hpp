#pragma once

#include <libminicpp/varitf.hpp>


class StableMatching : public Constraint
{
    // Constraint private data structures
    protected:
        std::vector<var<int>::Ptr> & _x;
        std::vector<var<int>::Ptr> & _y;
        std::vector<std::vector<int>> const & _xpl;
        std::vector<std::vector<int>> const & _ypl;
        int *_xPy;
        int *_yPx;
        int _n;
        std::vector<trail<int>> _yub;
        std::vector<trail<int>> _xlb;

    // Constraint methods
    public:
        StableMatching(std::vector<var<int>::Ptr> & m, std::vector<var<int>::Ptr> & w, std::vector<std::vector<int>> const & pm, std::vector<std::vector<int>> const & pw);
        void post() override;
        void propagate() override;
    
    protected:
        void buildReverseMatrix(std::vector<std::vector<int>> zpl, int *zPz);
};


