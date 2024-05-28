#include <stdint.h>
#include <stdio.h>

//const uint32_t UNS_ONE = 1;

__device__ int getBit2(uint32_t* bitmap, int index){
    int offset = index % 32;
    return (bitmap[index/32] << offset) >> (sizeof (int)*8 - 1);
}

__device__ int getDomainBit2(uint32_t* bitmap, int row, int column, int n){
    return getBit2(bitmap,row*n+column);
}

__device__ void delBit(uint32_t* bitmap, int index){
    int offset = index % 32;
    if ((bitmap[index/32] << offset) >> (sizeof (int)*8 - 1) != 0){
        //bitwise and not
        atomicAnd(&bitmap[index/32],~((UNS_ONE<< (sizeof (int)*8 - 1)) >> offset));
    }
}

__device__ void delDomainBit(uint32_t* bitmap, int row, int column, int n){
    delBit(bitmap,row*n+column);
}
