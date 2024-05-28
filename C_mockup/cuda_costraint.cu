#include <stdio.h>
#include "utils/cuda_domain_functions.cu"

__global__ void my_kernel(int n, int* xpl, int* ypl, int* xPy, int* yPx, uint32_t* x_domain, uint32_t* y_domain, int* array_mod_men, int* array_mod_women, int* array_min_mod_men, int* stack_mod_men, int* stack_mod_women, int* length_men_stack, int* length_women_stack, int* stack_mod_min_men, int* length_min_men_stack, int* old_min_man, int* old_max_man, int* old_min_woman, int* old_max_woman){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    //closes redundant threads
    if (id>= *length_men_stack + *length_women_stack){
        return;
    }
    //gets person associated with thread and picks the correct data structures
    int person, other_person, other_index, temp;
    int is_man = id < *length_men_stack;
    int *old_min, *old_max, *other_zPz, *zpl, *other_array_mod, *length_stack;
    uint32_t *person_domain, *other_domain;
    if(is_man){
        person = stack_mod_men[id];
        old_min =  old_min_man;
        old_max = old_max_man;
        person_domain = x_domain;
        other_domain = y_domain;
        zpl = xpl;
        other_zPz = yPx;
        other_array_mod = array_mod_women;
        length_stack = length_men_stack;
    } else {
        person = stack_mod_women[id - *length_men_stack];
        old_min = old_min_woman;
        old_max = old_max_woman;
        person_domain = y_domain;
        other_domain = x_domain;
        zpl = ypl;
        other_zPz = xPy;
        other_array_mod = array_mod_men;
        length_stack = length_women_stack;
    }

    //scans the domain, looking for removed values
    for(int i=old_min[person]; i<=old_max[person];i++){
        if(getDomainBit2(person_domain,person,i,n)==0){//this bit is 0
            other_person = zpl[person*n+i];
            if(getDomainBit2(other_domain,other_person,other_zPz[other_person*n+person],n)){//==1 other person's domain must be updated
                other_index = other_zPz[other_person*n+person];
                delDomainBit(other_domain,other_person,other_index,n);
                if(!atomicExch(&(other_array_mod[other_person]),1)){//it wasn't marked as modified (it was 0)
                    temp = atomicAdd(length_stack,1);
                    other_array_mod[temp] = other_person;
                    if(!is_man && old_min_man[other_person]==other_index){//updates array_min_mod_men if other_person is a man and the min was just removed
                        array_min_mod_men[other_person] = 1;
                        temp = atomicAdd(length_min_men_stack,1);
                        stack_mod_min_men[temp]=other_person;
                    }
                }
            }
        }
    }
}