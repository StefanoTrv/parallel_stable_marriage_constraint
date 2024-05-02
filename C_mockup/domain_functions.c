#include <stdint.h>

int getBit(int32_t* bitmap, int index){
    int offset = index % 32;
    return (bitmap[index/32] << offset) >> (sizeof (int) - 1);
}

int getDomainBit(int32_t* bitmap, int row, int column, int n){
    return getBit(bitmap,row*n+column);
}

void setBit(int32_t* bitmap, int index, int val){
    int offset = index % 32;
    if ((bitmap[index/32] << offset) >> (sizeof (int) - 1) != val){
        if (val==0){
            bitmap[index/32] = bitmap[index/32] - ((1<< (sizeof (int) - 1)) >> offset);
        }else{
            bitmap[index/32] = bitmap[index/32] + ((1<< (sizeof (int) - 1)) >> offset);
        }
    };
}

void setDomainBit(int32_t* bitmap, int row, int column, int n, int val){
    setBit(bitmap,row*n+column,val);
}

int getMin(int n, int32_t domain, int index){
    for(int i=0; i<n;i++){
        if(getDomainBit(domain,index,i,n)){
            return i;
        }
    }
    printf("getMin Error");
    return -1;
}

int getMax(int n, int32_t domain, int index){
    for(int i=n-1; i>=0;i--){
        if(getDomainBit(domain,index,i,n)){
            return i;
        }
    }
    printf("getMax Error");
    return -1;
}

int getVal(int n, int32_t domain, int index){
    int m = getMax(n,domain,index);
    if(m==getMin(n,domain,index)){
        return m;
    } else {
        printf("getVal Error");
        return -1;
    }
}

void setMax(int n, int32_t domain, int index, int a){
    if (a >= n){
        print("setMax error");
    }
    int old_max = getMax(n,domain,index);
    if(a<old_max){
        for(int i=a+1; i<=old_max;i++){
            setDomainBit(domain,index,i,n,0);
        }
    }
}

void setVal(int n, int32_t domain, int index, int a){
    if (a >= n){
        print("setVal error");
    }
    for(int i=0; i<n;i++){
        if (i!=a){
            setDomainBit(domain,index,i,n,0);
        }else{
            setDomainBit(domain,index,i,n,1);
        }
    }
}

void remVal(int n, int32_t domain, int index, int a){
    if (a >= n){
        print("setVal error");
    }
    setDomainBit(domain,index,a,n,0);
}