#include "stable_matching.cuh"

StableMatchingGPU::StableMatchingGPU(std::vector<var<int>::Ptr> & m, std::vector<var<int>::Ptr> & w, std::vector<std::vector<int>> const & mpl, std::vector<std::vector<int>> const & wpl) :
    Constraint(m[0]->getSolver()), _x(m), _y(w), _xpl(mpl), _ypl(wpl)
{
    setPriority(CLOW);

    // Get the size of the problem instance
    _n = static_cast<int>(_x.size());

    // Build inverse matrices
    _xPy = (int *)malloc(_n * _n * sizeof(int));
    _yPx = (int *)malloc(_n * _n * sizeof(int));
    buildReverseMatrix(_xpl,_xPy);
    buildReverseMatrix(_ypl,_yPx);

    //TODO
}

void StableMatchingGPU::post(){
    for (auto const & v : _x)
    {
        v->propagateOnDomainChange(this);
    }

    for (auto const & v : _y)
    {
        v->propagateOnDomainChange(this);
    }

    //TODO
}

void StableMatchingGPU::propagate(){
    //TODO
}

void StableMatchingGPU::buildReverseMatrix(std::vector<std::vector<int>> zpl, int *zPz){
    for(int i=0;i<_n;i++){
        for(int j=0;j<_n;j++){
            zPz[i*_n+zpl[i][j]]=j;
        }
    }
}