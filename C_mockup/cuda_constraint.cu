#include <stdio.h>
#include "utils/cuda_domain_functions.cu"

__constant__ uint32_t ALL_ONES = 4294967295;

// f1: removes from the women's domains the men who don't have that woman in their list (domain) anymore, and vice versa
// Modifies only the domains
__global__ void make_domains_coherent(int n, int* xpl, int* ypl, int* xPy, int* yPx, uint32_t* x_domain, uint32_t* y_domain, int* stack_mod_men, int* stack_mod_women, int length_men_stack, int length_women_stack, int* stack_mod_min_men, int* length_min_men_stack, int* old_min_men, int* old_max_men, int* old_min_women, int* old_max_women){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    //closes redundant threads
    if (id>= length_men_stack + length_women_stack){
        //printf("Returning %i\n",id);
        return;
    }
    //printf("Continuing %i\n",id);
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

// f2: applies the stable marriage constraint
// Modifies old_min_men, max_women and x_domain
__global__ void apply_sm_constraint(int n, int* xpl, int* ypl, int* xPy, int* yPx, uint32_t* x_domain, uint32_t* y_domain, int* array_min_mod_men, int* stack_mod_min_men, int* length_min_men_stack, int* new_stack_mod_min_men, int* new_length_min_men_stack, int* old_min_men, int* max_men, int* max_women, int* warp_counter){
    __shared__ int flag; // will be equal to *new_length_min_men_stack in the last warp, 0 in every other warp

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    //closes redundant threads
    //if (id>= *length_min_men_stack){
    //    return;
    //}

    // These values will be used later
    int lane_id = threadIdx.x % 32;
    int warpsPerBlock = (blockDim.x + 31) / 32;
    int warpCount = warpsPerBlock * gridDim.x; // gridDim.x = block size
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
        while(id < *length_min_men_stack){//Avoids memory access errors while keeping all warps active
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
            if (currentCount + 1 >= warpCount){ //greater for when it's not the first re-run
                //Using *new_length_min_men_stack causes termination when there's an empty domain and facilitates the reset of the data structures
                flag = *new_length_min_men_stack; 
            } else {
                flag = 0;
            }
        }
        __syncwarp();
        //If it's not the last active warp, there are no new free men, or an empty domain was found, it returns
        if (flag <= 0){
            return;
        }
        //If there are too many newly freed men, it returns
        if (flag > 32){
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

        //if (lane_id==0){
        //    *new_length_min_men_stack = flag;
        //}
        //return;

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