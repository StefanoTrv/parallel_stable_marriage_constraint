#include <stdint.h>
#include <stdio.h>

const uint32_t UNS_ONE = 1;

int getBit(uint32_t* bitmap, int index){
    int offset = index % 32;
    return (bitmap[index/32] << offset) >> (sizeof (int)*8 - 1);
}

int getDomainBit(uint32_t* bitmap, int row, int column, int n){
    return getBit(bitmap,row*n+column);
}

void setBit(uint32_t* bitmap, int index, int val){
    int offset = index % 32;
    if ((bitmap[index/32] << offset) >> (sizeof (int)*8 - 1) != val){
        if (val==0){
            //bitwise and not
            bitmap[index/32] = bitmap[index/32] & ~((UNS_ONE<< (sizeof (int)*8 - 1)) >> offset);
        }else{
            //bitwise or
            bitmap[index/32] = bitmap[index/32] | ((UNS_ONE<< (sizeof (int)*8 - 1)) >> offset);
        }
    }
}

void setDomainBit(uint32_t* bitmap, int row, int column, int n, int val){
    setBit(bitmap,row*n+column,val);
}

int getMin(int n, uint32_t* domain, int index){
    for(int i=0; i<n;i++){
        if(getDomainBit(domain,index,i,n)){
            return i;
        }
    }
    //printf("getMin Error");
    return -1;
}

int getMax(int n, uint32_t* domain, int index){
    for(int i=n-1; i>=0;i--){
        if(getDomainBit(domain,index,i,n)){
            return i;
        }
    }
    //printf("getMax Error");
    return -1;
}

int getVal(int n, uint32_t* domain, int index){
    int m = getMax(n,domain,index);
    if(m==getMin(n,domain,index)){
        return m;
    } else {
        printf("getVal Error");
        return -1;
    }
}

void setMax(int n, uint32_t* domain, int index, int a){
    if (a >= n){
        printf("setMax error");
    }
    int old_max = getMax(n,domain,index);
    for(int i=a+1; i<=old_max;i++){
        setDomainBit(domain,index,i,n,0);
    }
}

void setVal(int n, uint32_t* domain, int index, int a){
    if (a >= n){
        printf("setVal error");
    }
    for(int i=0; i<n;i++){
        if (i!=a){
            setDomainBit(domain,index,i,n,0);
        }else{
            setDomainBit(domain,index,i,n,1);
        }
    }
}

void remVal(int n, uint32_t* domain, int index, int a){
    if (a >= n){
        printf("setVal error");
    }
    setDomainBit(domain,index,a,n,0);
}

/*int main() {
    // Example usage and testing
    int n = 4;
    uint32_t domain = 0xFFFFFFFF; // Initialize domain with all bits set to 1
    int index = 0;

    // Testing getBit function
    printf("Bit at index %d: %d\n", index, getBit(&domain, index));

    // Testing getDomainBit function
    int row = 0, column = 0;
    printf("Domain bit at row %d, column %d: %d\n", row, column, getDomainBit(&domain, row, column, n));

    // Testing setBit function
    int val = 0;
    setBit(&domain, index, val);
    printf("Bit at index %d set to %d\n", index, val);
    printf("New value of domain: %u\n", domain);

    // Testing setDomainBit function
    int new_val = 1;
    setDomainBit(&domain, row, column, n, new_val);
    printf("Domain bit at row %d, column %d set to %d\n", row, column, new_val);
    printf("New value of domain: %u\n", domain);

    // Testing getMin function
    printf("Minimum value in domain: %d\n", getMin(n, &domain, index));

    // Testing getMax function
    printf("Maximum value in domain: %d\n", getMax(n, &domain, index));

    // Testing getVal function
    printf("Value in domain: %d\n", getVal(n, &domain, index));

    // Testing setMax function
    int new_max = 2;
    setMax(n, &domain, index, new_max);
    printf("Maximum value set to %d\n", new_max);
    printf("New value of domain: %u\n", domain);

    // Testing setVal function
    int new_value = 3;
    setVal(n, &domain, index, new_value);
    printf("Value set to %d\n", new_value);
    printf("New value of domain: %u\n", domain);

    // Testing remVal function
    remVal(n, &domain, index, new_value);
    printf("Value removed\n");
    printf("New value of domain: %u\n", domain);

    return 0;
}*/