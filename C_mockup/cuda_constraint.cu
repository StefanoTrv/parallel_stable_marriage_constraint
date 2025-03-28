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
        if(getDomainBit2(person_domain,person,i,n)==0){//this bit is 0
            other_person = zpl[person*n+i];
            if(getDomainBit2(other_domain,other_person,other_zPz[other_person*n+person],n)){//==1 other person's domain must be updated
                other_index = other_zPz[other_person*n+person];
                delDomainBit(other_domain,other_person,other_index,n);
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
__global__ void apply_sm_constraint(int n, int* xpl, int* ypl, int* xPy, int* yPx, uint32_t* x_domain, uint32_t* y_domain, int* array_min_mod_men, int* stack_mod_min_men, int* length_min_men_stack, int* new_stack_mod_min_men, int* new_length_min_men_stack, int* old_min_men, int* max_men, int* max_women){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    //closes redundant threads
    if (id>= *length_min_men_stack){
        return;
    }
    //printf("max_women[%i]=%i\n",id,max_women[id]);

    //finds man assigned to this thread
    int m = stack_mod_min_men[id];
    //printf("m for thread %i is: %i\n",id,m);

    //the variables named *_val represent the value of some person in the domain of another specific person of the opposite sex
    int w_index, w;
    int p_val, m_val;
    int succ_val, succ;
    int m_ith, w_val;

    //the thread cycles as long as it has a man assigned to it
    while(1){
        //finds the first woman remaining in m's domain/list
        w_index = old_min_men[m];
        //printf("w_index for man %i (thread %i): %i\n", m, id, w_index);
        if(w_index>max_men[m]){//empty domain
            old_min_men[m]=n; //not needed in the real constraint
            //printf("EMPTY DOMAIN\n");
            return;
        }else if(getDomainBit2(x_domain,m,w_index,n)){//value in domain
            //printf("new w_index for man %i (thread %i): %i\n", m, id, w_index);
            w = xpl[m*n+w_index];

            m_val = yPx[w*n+m];

            //atomic read-and-write of max_women[w]
            p_val = atomicMin(max_women+w, m_val);
            //printf("New max for woman %i is %i (thread %i)\n",w,(p_val<m_val) ? p_val : m_val,id);
            //printf("man %i is proposing to woman %i (with index %i) (thread %i)\n", m, w, p_val, id);
            //printf("p_val for man %i (thread %i): %i\n", m, id, p_val);
            //printf("m_val for man %i (thread %i): %i\n", m, id, m_val);

            if(m_val > p_val){//w prefers p to m
                //printf("Deleting woman %i (with index %i) from domain of man %i (thread %i), because the woman declined.\n",w,w_index,m,id);
                old_min_men[m]=w_index+1; //atomicMax could be used, but it would very rarely make a difference
                //printf("New old_min_men for man %i is %i\n",m,w_index+1);
                //printf("Caso1 for man %i (thread %i)\n", m, id);
                //continue;//continues with the same m
            } else if(p_val==m_val){//w is already with m
                //printf("Caso2 for man %i (thread %i): RETURNING\n", m, id);

                return;//the thread has no free man to find a woman for
            } else {//m_val<p_val, that is w prefers m to p
                succ_val = m_val + 1;
                while(succ_val<=p_val){
                    succ = ypl[w*n+succ_val];
                    delDomainBit(x_domain,succ,xPy[succ*n+w],n);
                    //printf("Deleting woman %i (with index %i) from domain of man %i (thread %i), because the man is a successor of %i.\n",w,xPy[succ*n+w],succ,id,m);
                    succ_val++;
                }
                //printf("Caso3 for man %i (thread %i). New man: %i\n", m, id, ypl[w*n+p_val]);
                m = ypl[w*n+p_val];
                //continue;//continues with m:=p
            }
        }else{//value not in domain
            old_min_men[m]=w_index+1; //atomicMax could be used, but it would very rarely make a difference
            //printf("New old_min_men for man %i is %i\n",m,w_index+1);
            w = xpl[m*n+w_index];
            m_val = yPx[w*n+m];
            //printf("Woman %i (index %i) is not in the domain of man %i (thread %i)\n",w,w_index,m,id);
            //atomic read-and-write of max_women[w]
            p_val = atomicMin(max_women+w, m_val-1);
            //printf("New max for woman %i is %i (thread %i)\n",w,((p_val<m_val-1) ? p_val : m_val-1),id);
            for(int i = m_val+1; i<=p_val; i++){//remove that woman from all the men that were removed from her domain (no need for m_val since the domains are coherent)
                if(getDomainBit2(y_domain,w,i,n)){//value wasn't already removed
                    m_ith=  ypl[w*n+i];
                    w_val = xPy[m_ith*n+w];
                    delDomainBit(x_domain,m_ith,w_val,n);
                    //printf("Deleted woman %i (value %i) from domain of man %i, because of 0 value in man %i (thread %i).\n",w,w_val,m_ith,m_val,id);
                }
            }
            if(p_val>m_val-1){//checks if the min of the last man has changed (the condition checks if the max of the woman changed)
                //printf("Thread %i checking if man %i needs to be updated later.\n",id,m_ith);
                m_ith=  ypl[w*n+p_val]; //necessary if a domain is empty
                w_val = xPy[m_ith*n+w]; //necessary if a domain is empty
                //printf("Value of p_val for thread %i is %i.\n",id,p_val);
                //printf("Value of m_ith for thread %i is %i.\n",id,m_ith);
                //printf("Value of w_val for thread %i is %i.\n",id,w_val);

                //marks the man as needing to be updated
                if(!atomicExch(&(array_min_mod_men[m_ith]),1)){ //atomic exchange to avoid duplicates (which could overflow the stack)
                    new_stack_mod_min_men[atomicAdd(new_length_min_men_stack,1)]=m_ith; //adds man to new stack
                    //printf("Thread %i found that man %i needs to be updated later.\n",id,m_ith);
                    //printf("Thread %i increased new_length_min_men_stack to %i for man %i\n",id,*new_length_min_men_stack,m_ith);
                }

            }
        }
        
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
        while(new_m>=0 && getDomainBit2(x_domain,id,new_m,n)==0){
            new_m--;
        }
    }
    old_max_men[id]=new_m;

    new_m=min_women[id];//old_min_women
    if(max_women[id]>=min_women[id]){
        while(new_m<n && getDomainBit2(y_domain,id,new_m,n)==0){
            new_m++;
        }
    }
    old_min_women[id]=new_m;

}