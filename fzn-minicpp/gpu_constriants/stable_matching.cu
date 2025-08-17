#include "stable_matching.cuh"
#include "error_handler.cuh"

const uint32_t UNS_ONE = 1;
__constant__ int d_n_SMP;
//__constant__ uint32_t ALL_ONES = 4294967295;

__host__ __device__ void get_block_number_and_dimension(int, int, int*, int*);
__global__ void make_domains_coherent(bool first_propagation, int n, int* xpl, int* ypl, int* xPy, int* yPx, uint32_t* x_domain, uint32_t* y_domain, int* array_min_mod_men, int* new_stack_mod_min_men, int* new_length_min_men_stack, int* stack_mod_men, int* stack_mod_women, int length_men_stack, int length_women_stack, int* stack_mod_min_men, int* length_min_men_stack, int* old_min_men, int* old_max_men, int* old_min_women, int* old_max_women, int* max_men, int* min_women, int* max_women, int* warp_counter);
__global__ void apply_sm_constraint(int n, int* xpl, int* ypl, int* xPy, int* yPx, uint32_t* x_domain, uint32_t* y_domain, int* array_min_mod_men, int* stack_mod_min_men, int* length_min_men_stack, int* new_stack_mod_min_men, int* new_length_min_men_stack, int* old_min_men, int* old_max_men, int* old_min_women, int* old_max_women, int* max_men, int* min_women, int* max_women, int* warp_counter);
__global__ void finalize_changes(int n, uint32_t* x_domain, uint32_t* y_domain, int* old_min_men, int* old_max_men, int* old_min_women, int* old_max_women, int* max_men, int* min_women, int* max_women);

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
    _x_domain = (uint32_t *)malloc(((_n * _n) / 32 + (_n % 32 != 0)) * 2 * sizeof(uint32_t));
    _y_domain = _x_domain + ((_n * _n) / 32 + (_n % 32 != 0));
    _xpl = (int *)malloc((_n * _n * 4 + _n * 10 + 2) * sizeof(int));
    _ypl = _xpl + _n * _n;
    _xPy = _ypl + _n * _n;
    _yPx = _xPy + _n * _n;
    _old_min_men = _yPx + _n * _n;
    _old_max_men = _old_min_men + _n;
    _old_min_women = _old_max_men + _n;
    _old_max_women = _old_min_women + _n;
    _max_men = _old_max_women + _n;
    _min_women = _max_men + _n;
    _max_women = _min_women + _n;
    _stack_mod_min_men = _max_women + _n;
    _length_min_men_stack = _stack_mod_min_men + _n;
    _new_length_min_men_stack = _length_min_men_stack + 1;
    _stack_mod_men = _new_length_min_men_stack + 1;
    _stack_mod_women = _stack_mod_men + _n;

    //Device memory allocation
    HANDLE_ERROR(cudaMalloc((void**)&_d_xpl, sizeof(int) * (_n * _n * 4 + _n * 12 + 3)));
    _d_ypl = _d_xpl + _n * _n;
    _d_xPy = _d_ypl + _n * _n;
    _d_yPx = _d_xPy + _n * _n;
    _d_old_min_men = _d_yPx + _n * _n;
    _d_old_max_men = _d_old_min_men + _n;
    _d_old_min_women = _d_old_max_men + _n;
    _d_old_max_women = _d_old_min_women + _n;
    _d_max_men = _d_old_max_women + _n;
    _d_min_women = _d_max_men + _n;
    _d_max_women = _d_min_women + _n;
    _d_stack_mod_min_men = _d_max_women + _n;
    _d_length_min_men_stack = _d_stack_mod_min_men + _n;
    _d_new_length_min_men_stack = _d_length_min_men_stack + 1;
    _d_stack_mod_men = _d_new_length_min_men_stack + 1;
    _d_stack_mod_women = _d_stack_mod_men + _n;
    _d_new_stack_mod_min_men = _d_stack_mod_women + _n;
    _d_array_min_mod_men = _d_new_stack_mod_min_men + _n;
    _d_warp_counter = _d_array_min_mod_men + _n;
    
    //Initialize trailable vectors
    for (int i = 0; i < _n; i++){
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
    *_new_length_min_men_stack = 0;
    for(int i=0;i<_n;i++){
        _old_min_men[i]=0;
        _old_min_women[i]=0;
        _old_max_men[i]=_n-1;
        _old_max_women[i]=_n-1;
    }

    //Get number of SMPs in the device
    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    _n_SMP = props.multiProcessorCount;
    cudaMemcpyToSymbol(d_n_SMP, &_n_SMP, sizeof(int));

    //Initializes _x_old_sizes and _y_old_sizes
    for (int i=0; i<_n; i++){
        _x_old_sizes.push_back(trail<int>(_x[0]->getSolver()->getStateManager(), _n));
        _y_old_sizes.push_back(trail<int>(_x[0]->getSolver()->getStateManager(), _n));
    }
}

void StableMatchingGPU::post(){
    int _length_men_stack, _length_women_stack;
    _length_men_stack = 0;
    _length_women_stack = 0;
    fillStacks(&_length_men_stack,&_length_women_stack);

    //Finds current maxes and mins
    for(int i=0; i<_n; i++){
        _min_women[i]=_y[i]->min();
        _max_men[i]=_x[i]->max();
        _max_women[i]=_y[i]->max();
    }

    HANDLE_ERROR(cudaMemcpyAsync(_d_xpl, _xpl, (_n * _n * 4 + _n * 10 + 2) * sizeof(int), cudaMemcpyHostToDevice, _stream));

    //Initializes the update_counters, which allow for baktracking identification
    _propagation_counter_trail = trail<int>(_x[0]->getSolver()->getStateManager(), 0);
    _propagation_counter = 1; // Mismatching values force dumping of all variables

    //Copy the domains
    int t1; //The person indexes are not needed in the post method (everything is copied)
    dumpDomainsToBitset(_x, _x_domain, _old_min_men, _old_max_men,_x_old_sizes,&t1,&t1);
    dumpDomainsToBitset(_y, _y_domain, _old_min_women, _old_max_women,_y_old_sizes,&t1,&t1);
    HANDLE_ERROR(cudaMemcpyAsync(_d_x_domain, _x_domain, ((_n * _n) / 32 + (_n % 32 != 0)) * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, _stream));

    /*
        Excute kernels
    */
    int n_threads = max(_length_men_stack + _length_women_stack,1); //At least one warp will be launched
    int n_blocks, block_size;
    get_block_number_and_dimension(n_threads,_n_SMP,&block_size,&n_blocks);
    HANDLE_ERROR(cudaMemsetAsync(_d_array_min_mod_men,0,sizeof(int)*(_n+1), _stream)); // sets to 0 d_array_min_mod_men and _d_warp_counter
    make_domains_coherent<<<n_blocks,block_size,0,_stream>>>(true, _n, _d_xpl, _d_ypl, _d_xPy, _d_yPx, _d_x_domain, _d_y_domain, _d_array_min_mod_men, _d_new_stack_mod_min_men, _d_new_length_min_men_stack, _d_stack_mod_men, _d_stack_mod_women, _length_men_stack, _length_women_stack, _d_stack_mod_min_men, _d_length_min_men_stack, _d_old_min_men, _d_old_max_men, _d_old_min_women, _d_old_max_women, _d_max_men, _d_min_women, _d_max_women, _d_warp_counter);

    /*
        Completed kernel execution (not yet synchronized)
    */

    //Update data structures and variables
    HANDLE_ERROR(cudaMemcpyAsync(_x_domain, _d_x_domain, ((_n * _n) / 32 + (_n % 32 != 0)) * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost, _stream));
    HANDLE_ERROR(cudaMemcpyAsync(_old_min_men, _d_old_min_men, sizeof(int) * _n * 4, cudaMemcpyDeviceToHost, _stream));
    cudaStreamSynchronize(_stream);
    updateHostData();

    //Set propagation condition for variables
    //After first propagation to avoid useless propagate() call
    for (auto const & v : _x){
        v->propagateOnDomainChange(this);
    }
    for (auto const & v : _y){
        v->propagateOnDomainChange(this);
    }

    //Sets the counters
    _propagation_counter_trail = 1;
    _propagation_counter = 1;

    //Updates _x_old_sizes and _y_old_sizes for the first propagation
    for (int i=0; i<_n; i++){
        _x_old_sizes[i] = _x[i]->size();
        _y_old_sizes[i] = _y[i]->size();
    }
}

void StableMatchingGPU::propagate(){
    //Prepare other data structures
    int _length_men_stack, _length_women_stack;
    _length_men_stack = 0;
    _length_women_stack = 0;
    *_length_min_men_stack = 0;
    *_new_length_min_men_stack = 0;
    fillStacks(&_length_men_stack, &_length_women_stack);
    if(_length_men_stack+_length_women_stack==0){ //no variable needs to be updated: quits immediately
        return;
    }
    for(int i=0; i<_n; i++){
        _min_women[i]=_y[i]->min();
        _max_men[i]=_x[i]->max();
        _max_women[i]=_y[i]->max();
    }

    // Restores the pointers to the mod_min_men stacks
    _d_stack_mod_min_men = _d_max_women + _n;
    _d_new_stack_mod_min_men = _d_stack_mod_women + _n;

    //copies old bounds arrays if needed, current bounds arrays, and stack_mod_min_men
    if(_propagation_counter_trail != _propagation_counter){ //has backtracked
        for(int i=0;i<_n;i++){
            _old_min_men[i]=_old_min_men_trail[i];
            _old_min_women[i]=_old_min_women_trail[i];
            _old_max_men[i]=_old_max_men_trail[i];
            _old_max_women[i]=_old_max_women_trail[i];
        }
        HANDLE_ERROR(cudaMemcpyAsync(_d_old_min_men, _old_min_men, (_n * 7 + *_length_min_men_stack) * sizeof(int), cudaMemcpyHostToDevice, _stream));
    }else{ //has not backtracked
        HANDLE_ERROR(cudaMemcpyAsync(_d_stack_mod_min_men, _stack_mod_min_men, (_n * 3 + *_length_min_men_stack) * sizeof(int), cudaMemcpyHostToDevice, _stream));
    }
    
    //copies length_min_men_stack, new_length_min_men_stack and stack_mod_men (if not empty)
    HANDLE_ERROR(cudaMemcpyAsync(_d_length_min_men_stack, _length_min_men_stack, (2 + _length_men_stack) * sizeof(int), cudaMemcpyHostToDevice, _stream));

    //copies stack_mod_women
    if(_length_women_stack>0){
        HANDLE_ERROR(cudaMemcpyAsync(_d_stack_mod_women, _stack_mod_women, (_length_women_stack) * sizeof(int), cudaMemcpyHostToDevice, _stream));
    }
    HANDLE_ERROR(cudaMemsetAsync(_d_array_min_mod_men,0,sizeof(int)*(_n+1), _stream)); // sets to 0 d_array_min_mod_men and _d_warp_counter

    //Copy domains to device
    int first_man, last_man, first_woman, last_woman;
    dumpDomainsToBitset(_x, _x_domain, _old_min_men, _old_max_men, _x_old_sizes, &first_man, &last_man);
    dumpDomainsToBitset(_y, _y_domain, _old_min_women, _old_max_women, _y_old_sizes, &first_woman, &last_woman);
    //Computes interval to be copied into the device
    int domain_offset, copy_size;
    if(first_man>=0){
        domain_offset = (first_man * _n) / 32;
    } else {
        domain_offset = ((_n * _n) / 32 + (_n % 32 != 0));
        domain_offset += (first_woman * _n) / 32;
    }
    if(last_woman>=0){
        copy_size = ((_n * _n) / 32 + (_n % 32 != 0)) * 2 - //Total size
                    domain_offset - //Offset
                    (_n - (last_woman +1)) * (_n / 32); //Unchanged last domains (words)
    } else {
        copy_size = ((_n * _n) / 32 + (_n % 32 != 0)) - //Men's domains' size
                    domain_offset - //Offset
                    (_n - (last_man +1)) * (_n / 32); //Unchanged last domains (words)
    }
    HANDLE_ERROR(cudaMemcpyAsync(_d_x_domain + domain_offset, _x_domain + domain_offset, copy_size * sizeof(uint32_t), cudaMemcpyHostToDevice, _stream));

    //Updates and increases counters
    _propagation_counter_trail += 1;
    _propagation_counter = _propagation_counter_trail;

    /*
        Excute kernels
    */
    int n_threads = _length_men_stack + _length_women_stack;
    int n_blocks, block_size;
    get_block_number_and_dimension(n_threads,_n_SMP,&block_size,&n_blocks);
    make_domains_coherent<<<n_blocks,block_size,0,_stream>>>(false, _n, _d_xpl, _d_ypl, _d_xPy, _d_yPx, _d_x_domain, _d_y_domain, _d_array_min_mod_men, _d_new_stack_mod_min_men, _d_new_length_min_men_stack, _d_stack_mod_men, _d_stack_mod_women, _length_men_stack, _length_women_stack, _d_stack_mod_min_men, _d_length_min_men_stack, _d_old_min_men, _d_old_max_men, _d_old_min_women, _d_old_max_women, _d_max_men, _d_min_women, _d_max_women, _d_warp_counter);

    /*
        Completed kernel execution (not yet synchronized)
    */

    //Update data structures and variables
    HANDLE_ERROR(cudaMemcpyAsync(_x_domain, _d_x_domain, ((_n * _n) / 32 + (_n % 32 != 0)) * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost, _stream));
    HANDLE_ERROR(cudaMemcpyAsync(_old_min_men, _d_old_min_men, sizeof(int) * _n * 4, cudaMemcpyDeviceToHost, _stream));
    cudaStreamSynchronize(_stream);
    updateHostData();

    //Updates old sizes
    for(int i=0; i<_n; i++){
        _x_old_sizes[i]=_x[i]->size();
        _y_old_sizes[i]=_y[i]->size();
    }
}

void StableMatchingGPU::fillStacks(int* _length_men_stack, int* _length_women_stack){
    int j;
    bool min_had_changed, size_had_changed;
    for (int i = 0; i < _n; i++){
        if (_x[i]->size() != _x_old_sizes[i]){ // if variable was modified (compares the sizes to avoid false positives given by changed())
            _stack_mod_men[*_length_men_stack] = i;
            (*_length_men_stack)++;
            if (_x[i]->min() != _old_min_men_trail[i]){ // if min is changed (this comparison avoids false positives given by changedMin())
                _stack_mod_min_men[*_length_min_men_stack] = i;
                (*_length_min_men_stack)++;
            }
            // Binding (optional for correctness)
            if (_x[i]->isBound()){ // binds the woman to whom the man is bound
                j = _xpl[i * _n + _x[i]->min()];
                if (j < i && _y[j]->size() == _y_old_sizes[j]){ // variable was not previously modified and won't be scanned again
                    _y[j]->assign(_yPx[j * _n + i]);
                    if (_y[j]->size() != _y_old_sizes[j])
                    { // makes sure the assignment really had an effect
                        _stack_mod_women[*_length_women_stack] = j;
                        (*_length_women_stack)++;
                    }
                }
                else{ // variable will be scanned later
                    _y[j]->assign(_yPx[j * _n + i]);
                }
            }
        }
        if (_y[i]->size() != _y_old_sizes[i]){ // if variable was modified (compares the sizes to avoid false positives given by changed())
            _stack_mod_women[*_length_women_stack] = i;
            (*_length_women_stack)++;
            // Binding (optional for correctness)
            if (_y[i]->isBound()){ // binds the man to whom the woman is bound
                j = _ypl[i * _n + _y[i]->min()];
                if (j <= i){ // variable won't be scanned again
                    min_had_changed = _x[j]->min() != _old_min_men_trail[j];
                    size_had_changed = _x[j]->size() != _x_old_sizes[j];
                    _x[j]->assign(_xPy[j * _n + i]);
                    if (!min_had_changed && _x[j]->min() != _old_min_men_trail[j]){ // min wasn't marked as changed but it has just been modified
                        _stack_mod_min_men[*_length_min_men_stack] = j;
                        (*_length_min_men_stack)++;
                    }
                    if (!size_had_changed && _x[j]->size() != _x_old_sizes[j]){ // variable wasn't modified before but it now is
                        _stack_mod_men[*_length_men_stack] = j;
                        (*_length_men_stack)++;
                    }
                }
                else{ // variable will be scanned later
                    _x[j]->assign(_xPy[j * _n + i]);
                }
            }
        }
    }
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

void StableMatchingGPU::dumpDomainsToBitset(std::vector<var<int>::Ptr> vars, uint32_t* dom, int* old_mins, int* old_maxes, std::vector<trail<int>> old_sizes, int* first_person, int* last_person){
    int starting_bit,ending_bit;
    int starting_word, ending_word;
    int var_min, var_max;
    uint32_t mask;
    bool has_backtracked = _propagation_counter_trail != _propagation_counter;
    *first_person = -1;
    *last_person = -1;
    for(int i=0; i<_n; i++){
        //_has_backtracked allows the bitset to be correct even after backtracking
        if(vars[i]->size()==old_sizes[i] && !has_backtracked){ //checks size to see if variable has changed
            continue;
        }

        //Computes interval to be copied to the GPU
        if(*first_person<0){
            *first_person = i;
        }
        *last_person = i;

        //Note: During the first dump, all the domains' bitsets will be completely overwritten.
        var_min = old_mins[i];
        var_max = old_maxes[i];

        starting_bit = _n * i + var_min;
        starting_word = starting_bit/32;
        ending_bit = _n * i + var_max;
        ending_word = ending_bit/32; 

        if (starting_word == ending_word) {
            // Clear bits within a single word
            mask = 0xFFFFFFFF >> ((31 - ending_bit) % 32) << ((31 - ending_bit) + starting_bit) % 32 >> (starting_bit % 32);
            dom[starting_word] &= ~mask;
        } else {
            // Clear bits in the first word
            mask = 0xFFFFFFFF << (starting_bit % 32) >> (starting_bit % 32);
            dom[starting_word] &= ~mask;
        
            // Clear full words in between
            for (int w = starting_word + 1; w < ending_word; w++) {
                dom[w] = 0;
            }
        
            // Clear bits in the last word
            mask = 0xFFFFFFFF >> ((31 - ending_bit) % 32) << ((31 - ending_bit) % 32);
            dom[ending_word] &= ~mask;
        }
        starting_bit = _n * i + vars[i]->min();
        vars[i]->dumpWithOffset(vars[i]->min(),vars[i]->max(),dom + (starting_bit / 32),starting_bit % 32);
    }
}

int StableMatchingGPU::getBitHost(uint32_t* bitmap, int index){
    int offset = index % 32;
    return (bitmap[index/32] << offset) >> (sizeof (uint32_t)*8 - 1);
}

int StableMatchingGPU::getDomainBitHost(uint32_t* bitmap, int row, int column){
    return getBitHost(bitmap,row*_n+column);
}

// Updates the _olds trailable vectors and the variables with the data from the device
// Supposes that the copy from the device has already been completed
void StableMatchingGPU::updateHostData(){
    for (int i = 0; i < _n; i++){ //_olds
        _old_max_women_trail[i]=_old_max_women[i];
        _old_min_women_trail[i]=_old_min_women[i];
        _old_max_men_trail[i]=_old_max_men[i];
        _old_min_men_trail[i]=_old_min_men[i];
    }

    for (int i=0; i<_n; i++){ //_x and _y
        _x[i]->removeBelow(_old_min_men[i]);
        _x[i]->removeAbove(_old_max_men[i]);
        for(int j=_old_min_men[i]+1; j<_old_max_men[i]; j++){
            if(!getDomainBitHost(_x_domain,i,j)){
                _x[i]->remove(j);
            }
        }

        _y[i]->removeBelow(_old_min_women[i]);
        _y[i]->removeAbove(_old_max_women[i]);
        for(int j=_old_min_women[i]+1; j<_old_max_women[i]; j++){
            if(!getDomainBitHost(_y_domain,i,j)){
                _y[i]->remove(j);
            }
        }
    }
}

__device__ int getBitCuda(uint32_t* bitmap, int index){
    int offset = index % 32;
    return (bitmap[index/32] << offset) >> (sizeof (uint32_t)*8 - 1);
}

__device__ int getDomainBitCuda(uint32_t* bitmap, int row, int column, int n){
    return getBitCuda(bitmap,row*n+column);
}

__device__ void delBitCuda(uint32_t* bitmap, int index){
    int offset = index % 32;
    if ((bitmap[index>>5] << offset) >> (sizeof (uint32_t)*8 - 1) != 0){//index>>5 == index/32
        //bitwise and not
        atomicAnd(&bitmap[index>>5],~((UNS_ONE<< (sizeof (uint32_t)*8 - 1)) >> offset));//index>>5 == index/32
    }
}

__device__ void delDomainBitCuda(uint32_t* bitmap, int row, int column, int n){
    delBitCuda(bitmap,row*n+column);
}

// f1: removes from the women's domains the men who don't have that woman in their list (domain) anymore, and vice versa
// Modifies only the domains
__global__ void make_domains_coherent(bool first_propagation, int n, int* xpl, int* ypl, int* xPy, int* yPx, uint32_t* x_domain, uint32_t* y_domain, int* array_min_mod_men, int* new_stack_mod_min_men, int* new_length_min_men_stack, int* stack_mod_men, int* stack_mod_women, int length_men_stack, int length_women_stack, int* stack_mod_min_men, int* length_min_men_stack, int* old_min_men, int* old_max_men, int* old_min_women, int* old_max_women, int* max_men, int* min_women, int* max_women, int* warp_counter){
    __shared__ bool flag; // True if this is the last warp to execute, False otherwise
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = threadIdx.x % 32;
    int warpsPerBlock = (blockDim.x + 31) / 32;
    int warpTotal = warpsPerBlock * gridDim.x; // gridDim.x = number of blocks
    int currentCount;

    //redundant threads do nothing
    if (id < length_men_stack + length_women_stack){
        //gets person associated with thread and picks the correct data structures
        int person, other_person, other_index, temp;
        int is_man = id < length_men_stack;
        int *old_min, *old_max, *other_zPz, *zpl;
        uint32_t *person_domain, *other_domain;
        if(is_man){
            person = stack_mod_men[id];
            old_min =  old_min_men;
            old_max = old_max_men;
            person_domain = x_domain;
            other_domain = y_domain;
            zpl = xpl;
            other_zPz = yPx;
        } else {
            person = stack_mod_women[id - length_men_stack];
            old_min = old_min_women;
            old_max = old_max_women;
            person_domain = y_domain;
            other_domain = x_domain;
            zpl = ypl;
            other_zPz = xPy;
        }

        //scans the domain, looking for removed values
        for(int i=old_min[person]; i<=old_max[person];i++){
            if(getDomainBitCuda(person_domain,person,i,n)==0){//this bit is 0
                other_person = zpl[person*n+i];
                if(getDomainBitCuda(other_domain,other_person,other_zPz[other_person*n+person],n)){//==1 other person's domain must be updated
                    other_index = other_zPz[other_person*n+person];
                    delDomainBitCuda(other_domain,other_person,other_index,n);
                    if(!is_man && old_min_men[other_person]==other_index){//updates stack_mod_min_men if other_person is a man and the min was just removed
                        temp = atomicAdd(length_min_men_stack,1);
                        stack_mod_min_men[temp]=other_person;
                    }
                }
            }
        }
    }

    __syncwarp();
    //The first thread of the last active warp launches f2
    if (lane_id==0){ //Checks whether it is the last thread
        currentCount = atomicAdd(warp_counter,1);
        flag = currentCount + 1 == warpTotal;
    }
    __syncwarp();
    if(!flag){ //Not the last warp, the threads return
        return;
    }
    if(first_propagation){ //The first propagation needs extra initialization
        for(int i=lane_id; i<n; i+=32){ //All threads cooperate to fill the stack
            stack_mod_min_men[i] = i;
        }
        if(lane_id==0){
            *length_min_men_stack = n;
        }
        __syncwarp();
    }
    if(lane_id==0){ //The first thread launches f2 (or f3 if f2 can be skipped)
        *warp_counter = 0;
        int block_size, n_blocks;
        if(*length_min_men_stack>0){
            get_block_number_and_dimension(*length_min_men_stack,d_n_SMP,&block_size,&n_blocks);
            apply_sm_constraint<<<n_blocks,block_size,0>>>(n, xpl, ypl, xPy, yPx, x_domain, y_domain, array_min_mod_men, stack_mod_min_men, length_min_men_stack, new_stack_mod_min_men, new_length_min_men_stack, old_min_men, old_max_men, old_min_women, old_max_women, max_men, min_women, max_women, warp_counter);
        } else {
            get_block_number_and_dimension(n,d_n_SMP,&block_size,&n_blocks);
            finalize_changes<<<n_blocks,block_size,0>>>(n, x_domain, y_domain, old_min_men, old_max_men, old_min_women, old_max_women, max_men, min_women, max_women);
        }
    }
}

// f2: applies the stable marriage constraint
// Modifies old_min_men, max_women and x_domain
__global__ void apply_sm_constraint(int n, int* xpl, int* ypl, int* xPy, int* yPx, uint32_t* x_domain, uint32_t* y_domain, int* array_min_mod_men, int* stack_mod_min_men, int* length_min_men_stack, int* new_stack_mod_min_men, int* new_length_min_men_stack, int* old_min_men, int* old_max_men, int* old_min_women, int* old_max_women, int* max_men, int* min_women, int* max_women, int* warp_counter){
    __shared__ int flag; // will be equal to *new_length_min_men_stack in the last warp, a negative number in every other warp

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // These values will be used later
    int lane_id = threadIdx.x % 32;
    int warpsPerBlock = (blockDim.x + 31) / 32;
    int warpTotal = warpsPerBlock * gridDim.x; // gridDim.x = number of blocks
    int currentCount;

    //the variables named *_val represent the value of some person in the domain of another specific person of the opposite sex
    int m;
    int w_index, w;
    int p_val, m_val;
    int succ_val, succ;
    int m_ith, w_val;

    //finds man assigned to this thread
    if(id<*length_min_men_stack){//If to avoid out of bounds access by still active redundant threads
        m = stack_mod_min_men[id];
    }
    
    //This external cycle allows the last warp to execute again, if appropriate
    while(1){
        //the thread cycles as long as it has a man assigned to it
        while(id < *length_min_men_stack){//Avoids memory access errors while keeping all warps active (instead of while true)
            //finds the first woman remaining in m's domain/list
            w_index = old_min_men[m];
            if(w_index>max_men[m]){//empty domain
                *new_length_min_men_stack = -n; //avoids further launches of f2 if there is an empty domain
                break;
            }else if(getDomainBitCuda(x_domain,m,w_index,n)){//value in domain
                w = xpl[m*n+w_index];

                m_val = yPx[w*n+m];

                //atomic read-and-write of max_women[w]
                p_val = atomicMin(max_women+w, m_val);

                if(m_val > p_val){//w prefers p to m
                    old_min_men[m]=w_index+1; //atomicMax could be used, but it would very rarely make a difference
                    //continue;//continues with the same m
                } else if(p_val==m_val){//w is already with m
                    break;//the thread has no free man to find a woman for
                } else {//m_val<p_val, that is w prefers m to p
                    succ_val = m_val + 1;
                    while(succ_val<=p_val){
                        succ = ypl[w*n+succ_val];
                        delDomainBitCuda(x_domain,succ,xPy[succ*n+w],n);
                        succ_val++;
                    }
                    m = ypl[w*n+p_val];
                    //continue;//continues with m:=p
                }
            }else{//value not in domain
                old_min_men[m]=w_index+1; //atomicMax could be used, but it would very rarely make a difference
                w = xpl[m*n+w_index];
                m_val = yPx[w*n+m];
                //atomic read-and-write of max_women[w]
                p_val = atomicMin(max_women+w, m_val-1);
                for(int i = m_val+1; i<=p_val; i++){//remove that woman from all the men that were removed from her domain (no need for m_val since the domains are coherent)
                    if(getDomainBitCuda(y_domain,w,i,n)){//value wasn't already removed
                        m_ith=  ypl[w*n+i];
                        w_val = xPy[m_ith*n+w];
                        delDomainBitCuda(x_domain,m_ith,w_val,n);
                    }
                }
                if(p_val>m_val-1){//checks if the min of the last man has changed (the condition checks if the max of the woman changed)
                    m_ith=  ypl[w*n+p_val]; //necessary if a domain is empty
                    w_val = xPy[m_ith*n+w]; //necessary if a domain is empty

                    //marks the man as needing to be updated
                    if(!atomicExch(&(array_min_mod_men[m_ith]),1)){ //atomic exchange to avoid duplicates (which could overflow the stack)
                        new_stack_mod_min_men[atomicAdd(new_length_min_men_stack,1)]=m_ith; //adds man to new stack
                    }

                }
            }

        }
        __syncwarp();
        //Checks if this warp is the last active warp
        if (lane_id==0){
            currentCount = atomicAdd(warp_counter,1);
            if (currentCount + 1 >= warpTotal){ //greater for when it's not the first re-run
                //Using *new_length_min_men_stack causes termination when there's an empty domain and facilitates the reset of the data structures
                flag = atomicAdd(new_length_min_men_stack,0); //the atomic operation ensures it is reading the up-to-date value
            } else {
                flag = -1; //-1 to be able to distinguish when f3 should be called or not
            }
        }
        __syncwarp();
        //If it's not the last active warp or an empty domain was found, it returns
        if (flag < 0){
            return;
        } else if (flag == 0){ //If there is no new free man, it launches f3 and returns
            if(lane_id==0){ //Only the first thread launches f3
                int block_size, n_blocks;
                get_block_number_and_dimension(n,d_n_SMP,&block_size,&n_blocks);
                finalize_changes<<<n_blocks,block_size,0>>>(n, x_domain, y_domain, old_min_men, old_max_men, old_min_women, old_max_women, max_men, min_women, max_women);
            }
            return;
        }

        //If there are too many newly freed men, it returns, but the first thread may launch a new grid
        if (flag > 32){
            for(int i=lane_id; i<n; i+=32){ //All threads cooperate to reset the array
                array_min_mod_men[i] = 0;
            }
            __syncwarp();
            if(lane_id==0){ //Only the first thread may launch a new grid
                int block_size, n_blocks;
                get_block_number_and_dimension(*new_length_min_men_stack,d_n_SMP,&block_size,&n_blocks);
                *length_min_men_stack = 0; //switches the two mod_min_men stacks
                *warp_counter = 0;
                apply_sm_constraint<<<n_blocks,block_size,0>>>(n,xpl,ypl,xPy,yPx,x_domain,y_domain, array_min_mod_men, new_stack_mod_min_men, new_length_min_men_stack, stack_mod_min_men, length_min_men_stack, old_min_men, old_max_men, old_min_women, old_max_women, max_men, min_women, max_women, warp_counter);
            }
            return;
        }
        //First thread sets the variable
        if (lane_id==0){
            *new_length_min_men_stack = 0;
            *length_min_men_stack = flag;
        }
        //The array is set in parallel
        for(int i=lane_id; i<n; i+=32){
            array_min_mod_men[i] = 0;
        }
        if (lane_id<flag){
            m = new_stack_mod_min_men[lane_id];
        }
        id = lane_id; //New id after all the other warps have finished
        __syncwarp();
    }
}

//f3: finalizes the changes in the domains and computes the new old_maxes and old_mins
// Modifies y_domain, old_max_women, old_max_men and old_min_women
__global__ void finalize_changes(int n, uint32_t* x_domain, uint32_t* y_domain, int* old_min_men, int* old_max_men, int* old_min_women, int* old_max_women, int* max_men, int* min_women, int* max_women){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    //closes redundant threads
    if (id>= n){
        return;
    }

    //finalizes women's domains
    /*int domain_offset = n * id;
    int first_bit_index = max_women[id]+1 + domain_offset; //need to add offset to find the domain of current woman, not the first one
    int last_bit_index = old_max_women[id] + domain_offset;
    int span = last_bit_index - first_bit_index + 1;
    int domain_index, n_bits, leftover_bits_in_word, offset;
    
    while(span>0){
        if(first_bit_index << (sizeof (int)*8 - 5) != 0 || span < 32){ //first_bit_index%32!=0, the last part of a word OR the first part of a word (beginning and/or end of area of interest)
            domain_index = first_bit_index>>5; //first_bit_index/32
            offset = first_bit_index%32; //offset of the first bit in the word
            leftover_bits_in_word = 32-offset; //the remaining bits from first_bit_index to the end of the word
            n_bits = leftover_bits_in_word<span ? leftover_bits_in_word : span; //how many bits to put in this word
            atomicAnd(&y_domain[domain_index],~((ALL_ONES<< (sizeof (int)*8 - n_bits)) >> offset)); //atomically deletes the appropriate bits of the word
            span-=n_bits; //marks some bits as added
            first_bit_index+=n_bits; //new index for the first bit that still hasn't been updated
        }else{//span>32, whole word can be written
            domain_index = first_bit_index>>5; //first_bit_index/32
            y_domain[domain_index]=0; //deletes whole word
            span-=32; //marks some bits as added
            first_bit_index+=32; //new index for the first bit that still hasn't been updated
        }
    }*/

    //updates old_max_men, old_min_women, old_max_women
    old_max_women[id]=max_women[id];

    int new_m=max_men[id];//old_max_men
    if(old_min_men[id]<=max_men[id]){
        while(new_m>=0 && getDomainBitCuda(x_domain,id,new_m,n)==0){
            new_m--;
        }
    }
    old_max_men[id]=new_m;

    new_m=min_women[id];//old_min_women
    if(max_women[id]>=min_women[id]){
        while(new_m<n && getDomainBitCuda(y_domain,id,new_m,n)==0){
            new_m++;
        }
    }
    old_min_women[id]=new_m;

}

/*
    Computes the appropriate block size and number of blocks based on the number of threads required and the number of SMPs
*/
__host__ __device__ void get_block_number_and_dimension(int n_threads, int n_SMP, int *block_size, int *n_blocks){
    if (n_threads/n_SMP >= 32){ //at least one warp per SMP
        *n_blocks = n_SMP;
        *block_size = (n_threads + *n_blocks - 1) / *n_blocks;
        // we need full warps
        if (*block_size<<(32-5)!=0){ // not divisible by 32
            *block_size = ((*block_size>>5) + 1) << 5; 
        }
    } else { //less than one warp per SMP
        *block_size = 32;
        *n_blocks = (n_threads + 31) / 32;
    }
}