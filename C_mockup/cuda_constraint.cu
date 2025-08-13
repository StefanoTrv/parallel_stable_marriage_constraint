#include <stdio.h>
#include "utils/cuda_domain_functions.cu"

__host__ __device__ void get_block_number_and_dimension(int, int, int*, int*);
__global__ void finalize_changes(int, uint32_t*, uint32_t*, int*, int*, int*, int*, int*, int*, int*);
__global__ void apply_sm_constraint(int, int*, int*, int*, int*, uint32_t*, uint32_t*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int);

__constant__ uint32_t ALL_ONES = 4294967295;
constexpr int max_grid_depth_const = 24; // max depth of dynamic parallelism. The first grid has level 0, the last max_grid_depth.
static_assert(max_grid_depth_const % 2 == 0, "max_grid_depth_const must be an even number"); // necessary to avoid errors with stack_mod_min_men and new_stack_mod_min_men
__constant__ int max_grid_depth = max_grid_depth_const;
__constant__ int d_n_SMP;

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
        //printf("Executing %i\n",id);
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
                        //printf("Temp for thread %i is %i\n",id,temp);
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
    if(lane_id==0){ //The first thread launches f2
        *warp_counter = 0;
        int block_size, n_blocks;
        get_block_number_and_dimension(*length_min_men_stack,d_n_SMP,&block_size,&n_blocks);
        apply_sm_constraint<<<n_blocks,block_size,0>>>(n, xpl, ypl, xPy, yPx, x_domain, y_domain, array_min_mod_men, stack_mod_min_men, length_min_men_stack, new_stack_mod_min_men, new_length_min_men_stack, old_min_men, old_max_men, old_min_women, old_max_women, max_men, min_women, max_women, warp_counter, 1);
    }

}

// f2: applies the stable marriage constraint
// Modifies old_min_men, max_women and x_domain
__global__ void apply_sm_constraint(int n, int* xpl, int* ypl, int* xPy, int* yPx, uint32_t* x_domain, uint32_t* y_domain, int* array_min_mod_men, int* stack_mod_min_men, int* length_min_men_stack, int* new_stack_mod_min_men, int* new_length_min_men_stack, int* old_min_men, int* old_max_men, int* old_min_women, int* old_max_women, int* max_men, int* min_women, int* max_women, int* warp_counter, int grid_depth){
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
                if(flag<0){
                    *length_min_men_stack = -1; //Tells host not to launch new phases (we don't know if the host will read this variable or the "new" version).
                }
            } else {
                flag = -1; //-1 to be able to distinguish when f3 should be called or not
            }
        }
        __syncwarp();
        //If it's not the last active warp or an empty domain was found, it returns
        if (flag < 0){
            return;
        } else if (flag == 0){ //If there is no new free man and max depth has not been reached, if launches f3 and returns
            if(lane_id==0 && grid_depth < max_grid_depth){ //Only the first thread launches f3
                *new_length_min_men_stack = -1; //Tells host not to launch new phases (we don't know which one will be read by the host).
                *length_min_men_stack = -1; //Tells host not to launch new phases (we don't know which one will be read by the host).
                int block_size, n_blocks;
                get_block_number_and_dimension(n,d_n_SMP,&block_size,&n_blocks);
                finalize_changes<<<n_blocks,block_size,0>>>(n, x_domain, y_domain, old_min_men, old_max_men, old_min_women, old_max_women, max_men, min_women, max_women);
            }
            return;
        }

        //If there are too many newly freed men, it returns, but the first thread may launch a new grid
        if (flag > 32){
            if (grid_depth < max_grid_depth){
                for(int i=lane_id; i<n; i+=32){ //All threads cooperate to reset the array
                    array_min_mod_men[i] = 0;
                }
                __syncwarp();
                if(lane_id==0){ //Only the first thread may launch a new grid
                    //printf("New internal launch at depth %i.\n",grid_depth+1);
                    int block_size, n_blocks;
                    get_block_number_and_dimension(*new_length_min_men_stack,d_n_SMP,&block_size,&n_blocks);
                    *length_min_men_stack = 0; //switches the two mod_min_men stacks
                    *warp_counter = 0;
                    apply_sm_constraint<<<n_blocks,block_size,0>>>(n,xpl,ypl,xPy,yPx,x_domain,y_domain, array_min_mod_men, new_stack_mod_min_men, new_length_min_men_stack, stack_mod_min_men, length_min_men_stack, old_min_men, old_max_men, old_min_women, old_max_women, max_men, min_women, max_women, warp_counter, grid_depth+1);
                }
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

    //printf("max_women[%i] = %i\n",id,max_women[id]);

    //finalizes women's domains
    int domain_offset = n * id;
    int first_bit_index = max_women[id]+1 + domain_offset; //need to add offset to find the domain of current woman, not the first one
    int last_bit_index = old_max_women[id] + domain_offset;
    int span = last_bit_index - first_bit_index + 1;
    int domain_index, n_bits, leftover_bits_in_word, offset;
    
    while(span>0){
        if(first_bit_index << (sizeof (int)*8 - 5) != 0 || span < 32){ //first_bit_index%32!=0, the last part of a word OR the first part of a word (beginning and/or end of area of interest)
            //printf("Deleting part of word for woman %i\n",id);
            domain_index = first_bit_index>>5; //first_bit_index/32
            offset = first_bit_index%32; //offset of the first bit in the word
            leftover_bits_in_word = 32-offset; //the remaining bits from first_bit_index to the end of the word
            n_bits = leftover_bits_in_word<span ? leftover_bits_in_word : span; //how many bits to put in this word
            //printf("Mask value for woman %i and n_bits %i and offset %i: %i\n",id,n_bits,offset,~((ALL_ONES<< (sizeof (int)*8 - n_bits)) >> offset));
            atomicAnd(&y_domain[domain_index],~((ALL_ONES<< (sizeof (int)*8 - n_bits)) >> offset)); //atomically deletes the appropriate bits of the word
            span-=n_bits; //marks some bits as added
            first_bit_index+=n_bits; //new index for the first bit that still hasn't been updated
        }else{//span>32, whole word can be written
            //printf("Deleting whole word for woman %i\n",id);
            domain_index = first_bit_index>>5; //first_bit_index/32
            y_domain[domain_index]=0; //deletes whole word
            span-=32; //marks some bits as added
            first_bit_index+=32; //new index for the first bit that still hasn't been updated
        }
    }

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
