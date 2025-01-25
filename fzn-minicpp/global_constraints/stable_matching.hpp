#pragma once

#include <libminicpp/varitf.hpp>
#include <queue>


class StableMatching : public Constraint
{
    //"Private" struct
    struct constraintCall{
    // 0: removeValue
    // 1: deltaMin
    // 2: deltaMax
    int function;

    int ij;
    // The man or woman interested

    int a;
    // The value (index) to be removed by removeValue

    int isMan;
    // True if removeValue is working on a man, False if it's working on a woman

    // Constructor
    constraintCall(int function, int ij, int a, int isMan)
        : function(function), ij(ij), a(a), isMan(isMan) {}
    };

    // Constraint private data structures
    protected:
        std::vector<var<int>::Ptr> _x;
        std::vector<var<int>::Ptr> _y;
        std::vector<std::vector<int>> const _xpl;
        std::vector<std::vector<int>> const _ypl;
        int *_xPy;
        int *_yPx;
        int _n;
        std::vector<trail<int>> _xub;
        std::vector<trail<int>> _xlb;
        std::vector<trail<int>> _yub;
        std::vector<trail<int>> _ylb;
        std::queue<constraintCall> _callQueue;

    // Constraint methods
    public:
        StableMatching(std::vector<var<int>::Ptr> & m, std::vector<var<int>::Ptr> & w, std::vector<std::vector<int>> const & pm, std::vector<std::vector<int>> const & pw);
        void post() override;
        void propagate() override;
    
    protected:
        void buildReverseMatrix(std::vector<std::vector<int>> zpl, int *zPz);
        void propagateOnQueue();
        void functionDispatcher();
        void removeValue(int i, int a, int isMan);
        void deltaMin(int i);
        void deltaMax(int j);
        void init();
};


