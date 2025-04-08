#include <stdio.h>
#include "utils/cuda_domain_functions.cu"

__global__ void cuda_stable_marriage_kernel(int n, int* xpl, int* ypl, int* xPy, int* yPx, uint32_t* x_domain, uint32_t* y_domain, int* old_min_men, int* max_women){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    //closes redundant threads
    if (id>= n){
        return;
    }

    //finds man assigned to this thread
    int m = id;

    //the variables named *_val represent the value of some person in the domain of another specific person of the opposite sex
    int w_index, w;
    int p_val, m_val;
    int succ_val, succ;

    //the thread cycles as long as it has a man assigned to it
    while(1){
        //finds the first woman remaining in m's domain/list
        w_index = old_min_men[m];
        if(w_index==n){//empty domain
            return;
        }else if(getDomainBit2(x_domain,m,w_index,n)){//value in domain
            w = xpl[m*n+w_index];

            m_val = yPx[w*n+m];

            //atomic read-and-write of max_women[w]
            p_val = atomicMin(max_women+w, m_val);

            if(m_val > p_val){//w prefers p to m
                old_min_men[m]=w_index+1; //atomicMax could be used, but it would very rarely make a difference
                //continue;//continues with the same m
            } else if(p_val==m_val){//w is already with m

                return;//the thread has no free man to find a woman for
            } else {//m_val<p_val, that is w prefers m to p
                succ_val = m_val + 1;
                while(succ_val<=p_val){
                    succ = ypl[w*n+succ_val];
                    delDomainBit(x_domain,succ,xPy[succ*n+w],n);
                    succ_val++;
                }
                m = ypl[w*n+p_val];
                //continue;//continues with m:=p
            }
        }else{//value not in domain
            old_min_men[m]=w_index+1; //atomicMax could be used, but it would very rarely make a difference
        }
        
    }
}