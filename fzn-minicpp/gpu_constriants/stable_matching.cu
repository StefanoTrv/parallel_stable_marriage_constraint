#include "stable_matching.cuh"
#include "error_handler.cuh"


void print_domains(int n, uint32_t* x_dom, uint32_t* y_dom) {
    int index, offset;
    // Men
    printf("\nMen domains:\n");
    printf("_\t");
    for (int i = 0; i < n; i++) {
        printf("%d\t", i);
    }
    for (int i = 0; i < n; i++) {
        printf("\n%d:\t", i);
        for (int j = 0; j < n; j++) {
            index = i*n+j;
            offset = index % 32;
            printf("%d\t", (x_dom[index/32] << offset) >> (sizeof (int)*8 - 1));
        }
        printf("\n");
    }
    
    printf("\n");

    // Women
    printf("\nWomen domains:\n");
    printf("_\t");
    for (int i = 0; i < n; i++) {
        printf("%d\t", i);
    }
    for (int i = 0; i < n; i++) {
        printf("\n%d:\t", i);
        for (int j = 0; j < n; j++) {
            index = i*n+j;
            offset = index % 32;
            printf("%d\t", (y_dom[index/32] << offset) >> (sizeof (int)*8 - 1));
        }
        printf("\n");
    }
            
    printf("\n\n");
}

StableMatchingGPU::StableMatchingGPU(std::vector<var<int>::Ptr> & m, std::vector<var<int>::Ptr> & w, std::vector<std::vector<int>> const & mpl, std::vector<std::vector<int>> const & wpl) :
    Constraint(m[0]->getSolver()), _x(m), _y(w), _xpl_vector(mpl), _ypl_vector(wpl)
{
    setPriority(CLOW);
    cudaStreamCreate(&_stream);

    // Get the size of the problem instance
    _n = static_cast<int>(_x.size());

    //Domain device memory allocation
    HANDLE_ERROR(cudaMalloc((void**)&_d_x_domain, ((_n * _n) / 32 + (_n % 32 != 0)) * 2 * sizeof(uint32_t)));
    _d_y_domain = _d_x_domain + (_n * _n) / 32 + (_n % 32 != 0);

    //Host memory allocation
    _x_domain = (uint32_t *)calloc(((_n * _n) / 32 + (_n % 32 != 0)) * 2, sizeof(uint32_t)); // calloc() because shortened domains may mean some bits are not written
    _y_domain = _x_domain + ((_n * _n) / 32 + (_n % 32 != 0));
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
    
    //Initialize trailable vectors
    for (int i = 0; i < 10; i  += 1){
        _old_max_women_trail.push_back(trail<int>(m[0]->getSolver()->getStateManager(), _n-1));
        _old_min_women_trail.push_back(trail<int>(m[0]->getSolver()->getStateManager(), 0));
        _old_max_men_trail.push_back(trail<int>(m[0]->getSolver()->getStateManager(), _n-1));
        _old_min_men_trail.push_back(trail<int>(m[0]->getSolver()->getStateManager(), 0));
    }

    //Prepares the data structures that won't be modified by post()
    copyPreferenceMatrix(_xpl_vector,_xpl);
    copyPreferenceMatrix(_ypl_vector,_ypl);
    buildReverseMatrix(_xpl_vector,_xPy);
    buildReverseMatrix(_ypl_vector,_yPx);
    *_length_min_men_stack = 0; //for f1 we pretend that it's empty, then we fill it before f2
    *_new_length_min_men_stack = 0; //TODO forse spostare
    for (int i=0;i<_n;i++){
        _stack_mod_men[i]=i;
        _stack_mod_women[i]=i;
    }
    for(int i=0;i<_n;i++){
        _old_min_men[i]=0;
        _old_min_women[i]=0;
        _old_max_men[i]=_n-1;
        _old_max_women[i]=_n-1;
    }

    //Copy the data structures that won't be modified by post()
    HANDLE_ERROR(cudaMemcpyAsync(_d_xpl, _xpl, (_n * _n * 4 + _n * 7) * sizeof(int), cudaMemcpyHostToDevice, _stream));

    //Get number of SMPs in the device
    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    _n_SMP = props.multiProcessorCount;
}

void StableMatchingGPU::post(){
    int _length_men_stack, _length_women_stack;
    _length_men_stack = _n;
    _length_women_stack = _n;

    //Finds current maxes and mins
    for(int i=0; i<_n; i++){
        _min_women[i]=_y[i]->min();
        _max_men[i]=_x[i]->max();
        _max_women[i]=_y[i]->max();
    }

    //Copies the remaining data structures
    HANDLE_ERROR(cudaMemcpyAsync(_d_max_men, _max_men, (_n * 3 + 2) * sizeof(int), cudaMemcpyHostToDevice, _stream));

    for (auto const & v : _x){
        v->propagateOnDomainChange(this);
    }
    for (auto const & v : _y){
        v->propagateOnDomainChange(this);
    }

    dumpDomainsToBitset(true, _x, _x_domain, _old_min_men, _old_max_men);
    dumpDomainsToBitset(true, _y, _y_domain, _old_min_women, _old_max_women);

    HANDLE_ERROR(cudaMemcpyAsync(_d_x_domain, _x_domain, ((_n * _n) / 32 + (_n % 32 != 0)) * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, _stream));

    print_domains(_n,_x_domain,_y_domain);

    //TODO call & update olds and vars
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

void StableMatchingGPU::copyPreferenceMatrix(std::vector<std::vector<int>> zpl_vec, int *zpl){
    for(int i=0;i<_n;i++){
        for(int j=0;j<_n;j++){
            zpl[i*_n+j]=zpl_vec[i][j];
        }
    }
}

void StableMatchingGPU::dumpDomainsToBitset(bool first_dump, std::vector<var<int>::Ptr> vars, uint32_t* dom, int* old_mins, int* old_maxes){
    int starting_bit;
    int var_min, var_max;
    for(int i=0; i<_n; i++){
        if(!(vars[i]->changed()) && !first_dump){
            continue;
        }

        /*
            During the first execution we use the current maxes and min, since variable domains may be shortened (not [0,n-1]).
            Since we initialized the rest of the domains to 0s, we can ignore the part of the domains outside these initial bounds.
        */
        if(first_dump){
            var_min = vars[i]->min();
            var_max = vars[i]->max();
        } else {
            var_min = old_mins[i];
            var_max = old_maxes[i];
        }

        starting_bit = _n * i + var_min;

        vars[i]->dumpWithOffset(var_min,var_max,dom + (starting_bit / 32),starting_bit % 32);
    }
}