#include "stable_matching.cuh"
#include "error_handler.cuh"

StableMatchingGPU::StableMatchingGPU(std::vector<var<int>::Ptr> & m, std::vector<var<int>::Ptr> & w, std::vector<std::vector<int>> const & mpl, std::vector<std::vector<int>> const & wpl) :
    Constraint(m[0]->getSolver()), _x(m), _y(w), _xpl_vector(mpl), _ypl_vector(wpl)
{
    setPriority(CLOW);

    // Get the size of the problem instance
    _n = static_cast<int>(_x.size());

    //Domain device memory allocation
    HANDLE_ERROR(cudaMalloc((void**)&_d_x_domain, ((_n * _n) / 32 + (_n % 32 != 0)) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&_d_y_domain, ((_n * _n) / 32 + (_n % 32 != 0)) * sizeof(uint32_t))); //TODO compattare
    //MISSING TRANSFER TODO

    //Host memory allocation
    int length_men_stack, length_women_stack;
    _xpl = (int *)malloc((_n * _n * 4 + _n * 10 + 2) * sizeof(int));
    _ypl = _xpl + (_n* _n);
    _xPy = _ypl + (_n* _n);
    _yPx = _xPy + (_n* _n);
    _stack_mod_men = _yPx + (_n* _n);
    _stack_mod_women = _stack_mod_men + _n;
    _stack_mod_min_men = _stack_mod_women + _n;
    _old_min_men = _stack_mod_min_men + _n;
    _old_max_men = _old_min_men + _n;
    _old_min_women = _old_max_men + _n;
    _old_max_women = _old_min_women + _n;
    _max_men = _old_max_women + _n;
    _min_women = _max_men + _n;
    _max_women = _min_women + _n;
    _length_min_men_stack = _max_women + _n;
    _new_length_min_men_stack = _length_min_men_stack + 1;

    //Device memory allocation
    HANDLE_ERROR(cudaMalloc((void**)&_d_xpl, sizeof(int) * (_n * _n * 4 + _n * 12 + 2)));
    _d_ypl = _d_xpl + _n * _n;
    _d_xPy = _d_ypl + _n * _n;
    _d_yPx = _d_xPy + _n * _n;
    _d_stack_mod_men = _d_yPx + _n * _n;
    _d_stack_mod_women = _d_stack_mod_men + _n;
    _d_stack_mod_min_men = _d_stack_mod_women + _n;
    _d_old_min_men = _d_stack_mod_min_men + _n;
    _d_old_max_men = _d_old_min_men + _n;
    _d_old_min_women = _d_old_max_men + _n;
    _d_old_max_women = _d_old_min_women + _n;
    _d_max_men = _d_old_max_women + _n;
    _d_min_women = _d_max_men + _n;
    _d_max_women = _d_min_women + _n;
    _d_length_min_men_stack = _d_max_women + _n;
    _d_new_length_min_men_stack = _d_length_min_men_stack + 1;
    _d_new_stack_mod_min_men = _d_new_length_min_men_stack + 1;
    _d_array_min_mod_men = _d_new_stack_mod_min_men + _n;

    //Prepares all the data structures
    buildReverseMatrix(_xpl_vector,_xPy);
    buildReverseMatrix(_ypl_vector,_yPx);
    
    //Initialize ylb, yub, xlb and xub
    for (int i = 0; i < 10; i  += 1){
        _old_max_women_trail.push_back(trail<int>(m[0]->getSolver()->getStateManager(), _n-1));
        _old_min_women_trail.push_back(trail<int>(m[0]->getSolver()->getStateManager(), 0));
        _old_max_men_trail.push_back(trail<int>(m[0]->getSolver()->getStateManager(), _n-1));
        _old_min_men_trail.push_back(trail<int>(m[0]->getSolver()->getStateManager(), 0));
    }

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