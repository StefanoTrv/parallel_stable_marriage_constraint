#include <stdint.h>

const uint32_t UNS_ONE = 1;

__device__ int getBit2(uint32_t* bitmap, int index){
    int offset = index % 32;
    return (bitmap[index/32] << offset) >> (sizeof (uint32_t)*8 - 1);
}

__device__ int getDomainBitCuda(uint32_t* bitmap, int row, int column, int n){
    return getBit2(bitmap,row*n+column);
}

__device__ void delBit(uint32_t* bitmap, int index){
    int offset = index % 32;
    if ((bitmap[index>>5] << offset) >> (sizeof (uint32_t)*8 - 1) != 0){//index>>5 == index/32
        //bitwise and not
        atomicAnd(&bitmap[index>>5],~((UNS_ONE<< (sizeof (uint32_t)*8 - 1)) >> offset));//index>>5 == index/32
    }
}

__device__ void delDomainBitCuda(uint32_t* bitmap, int row, int column, int n){
    delBit(bitmap,row*n+column);
}
