#include <stdio.h>
#include "utils/cuda_domain_functions.cu"

__global__ void my_kernel(int n, int* d_xpl, int* d_ypl, int* d_xPy, int* d_yPx, uint32_t* d_x_domain, uint32_t* d_y_domain){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id>= n*n){
        return;
    }
    if (id%2==0){
        delBit(d_x_domain,id);
        if(getBit2(d_x_domain,id)!=0){
            printf("Error at %i, should be 0",id);
        }
    }else{
        if(getBit2(d_x_domain,id)!=1){
            printf("Error at %i, should be 1",id);
        }
    }

}