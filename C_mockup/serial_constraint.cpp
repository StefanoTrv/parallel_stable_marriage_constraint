#include "domain_functions.c"
#include <queue>

struct constraintCall{
    // 0: removeValue
    // 1: deltaMin
    // 2: deltaMax
    // 3: inst
    int function;

    int ij;
    // The man or woman interested

    int a;
    // The value (index) to be removed by removeValue

    int isMan;
    // True if removeValue is working on a man, False if it's working on a woman

    // Constructor
    constraintCall(int function, int ij, int a, int isMan)
        : function(function), ij(ij), a(a), isMan(isMan) {}
};

void inst(int i, int n, int isMan, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx, int* xlb, int* yub, std::queue<constraintCall> *queue){
    int j;
    if(isMan){
        for(int k=xlb[i]; k<getMin(n,x_domain,i);k++){
            j = xpl[i*n+k];
            if (getMax(n,y_domain,j)>yPx[j*n+i]-1){
                setMax(n,y_domain,j,yPx[j*n+i]-1);
                //Adds deltaMax to queue
                queue->push(constraintCall(2,j,0,0));
            }
        }
        j = xpl[i*n+getVal(n,x_domain,i)];
        if(getVal(n,y_domain,j)!=yPx[j*n+i]){
            setMax(n,y_domain,j,yPx[j*n+i]);
            setMin(n,y_domain,j,yPx[j*n+i]);
            //Adds inst to queue
            queue->push(constraintCall(3,j,0,0));
        }
        for(int k = getVal(n,x_domain,i)+1; k<n; k++){
            j = xpl[i*n+k];
            if(getDomainBit(y_domain,j,yPx[j*n+i],n)){
                if(getMax(n,y_domain,j)==yPx[j*n+i]){
                    //Adds deltaMax to queue
                    queue->push(constraintCall(2,j,0,0));
                }
                remVal(n,y_domain,j,yPx[j*n+i]);
            }
        }
        xlb[i] = getMin(n,x_domain,i);
    } else {
        for(int k=0; k<getMin(n,y_domain,i);k++){
            j = ypl[i*n+k];
            if(getDomainBit(x_domain,j,xPy[j*n+i],n)){
                if(getMin(n,x_domain,j)==xPy[j*n+i]){
                    //Adds deltaMin to queue
                    queue->push(constraintCall(1,j,0,0));
                }
                remVal(n,x_domain,j,xPy[j*n+i]);
            }
        }
        j = ypl[i*n+getVal(n,y_domain,i)];
        if(getVal(n,x_domain,j)!=xPy[j*n+i]){
            setMax(n,x_domain,j,xPy[j*n+i]);
            setMin(n,x_domain,j,xPy[j*n+i]);
            //Adds inst to queue
            queue->push(constraintCall(3,j,0,1));
        }
        for(int k = getVal(n,y_domain,i)+1; k<=yub[i]; k++){
            j = ypl[i*n+k];
            if(getDomainBit(x_domain,j,xPy[j*n+i],n)){
                if(getMin(n,x_domain,j)==xPy[j*n+i]){
                    //Adds deltaMin to queue
                    queue->push(constraintCall(1,j,0,0));
                }
                remVal(n,x_domain,j,xPy[j*n+i]);
            }
        }
        yub[i]=getMax(n,y_domain,i);
    }
}

void removeValue(int i, int a, int isMan, int n, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx, std::queue<constraintCall> *queue){
    if (isMan){
        int j = xpl[i*n+a];
        if (getDomainBit(y_domain,j,yPx[j*n+i],n)){
            if(getMax(n,y_domain,j)==yPx[j*n+i]){
                //Adds deltaMax to queue
                queue->push(constraintCall(2,j,0,0));
            }
            remVal(n,y_domain,j,yPx[j*n+i]);
        }
    } else {
        int j = ypl[i*n+a];
        if (getDomainBit(x_domain,j,xPy[j*n+i],n)){
            if(getMin(n,x_domain,j)==xPy[j*n+i]){
                //Adds deltaMin to queue
                queue->push(constraintCall(1,j,0,0));
            }
            remVal(n,x_domain,j,xPy[j*n+i]);
        }
    }
}

void deltaMin(int i, int n, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx, int* xlb, std::queue<constraintCall> *queue){
    if (getMin(n,x_domain,i)==n) return; //Domain is empty, no reason to continue
    int j = xpl[i*n+getMin(n,x_domain,i)];
    if (getMax(n,y_domain,j)>yPx[j*n+i]){
        setMax(n,y_domain,j,yPx[j*n+i]);
        //Adds deltaMax to queue
        queue->push(constraintCall(2,j,0,0));
    }
    for(int k=xlb[i]; k<getMin(n,x_domain,i);k++){
        j = xpl[i*n+k];
        if (getMax(n,y_domain,j)>yPx[j*n+i]-1){
            setMax(n,y_domain,j,yPx[j*n+i]-1);
            //Adds deltaMax to queue
            queue->push(constraintCall(2,j,0,0));
        }
    }
    xlb[i] = getMin(n,x_domain,i);
}

void deltaMax(int j, int n, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx, int* yub, std::queue<constraintCall> *queue){
    int i;
    for(int k=getMax(n,y_domain,j)+1;k<=yub[j];k++){
        i = ypl[j*n+k];
        if(getDomainBit(x_domain,i,xPy[i*n+j],n)){
            if(getMin(n,x_domain,i)==xPy[i*n+j]){
                //Adds deltaMin to queue
                queue->push(constraintCall(1,i,0,0));
            }
            remVal(n,x_domain,i,xPy[i*n+j]);
        }
    }
    yub[j]=getMax(n,y_domain,j);
}

void init(int n, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx, int* xlb, std::queue<constraintCall> *queue){
    for(int i=0;i<n;i++){
        deltaMin(i,n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb,queue);
    }
}