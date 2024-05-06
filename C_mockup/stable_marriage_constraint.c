#include "domain_functions.c"

void inst(int i, int n, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx){
    int j;
    for(int k=0;k<getVal(n,x_domain,i);k++){
        j = xpl[i*n+k];
        setMax(n,y_domain,j,yPx[j*n+i]-1);
    }

    j = xpl[i*n+getVal(n,x_domain,i)];
    setVal(n,y_domain,j,yPx[j*n+i]);

    for(int k=getVal(n,x_domain,i)+1;k<n;k++){
        j = xpl[i*n+k];
        remVal(n,y_domain,j,yPx[j*n+i]);
    }
}

void removeValue(int i, int a, int n, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx){
    int j = xpl[i*n+a];
    remVal(n,y_domain,j,yPx[j*n+i]);
}

void deltaMin(int i, int n, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx, int* xlb){
    int j = xpl[i*n+getMin(n,x_domain,i)];
    setMax(n,y_domain,j,yPx[j*n+i]);
    for(int k=xlb[i]; k<getMin(n,x_domain,i);k++){
        j = xpl[i*n+k];
        setMax(n,y_domain,j,yPx[j*n+i]-1);
    }
    xlb[i] = getMin(n,x_domain,i);
}

void deltaMax(int j, int n, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx, int* yub){
    int i;
    for(int k=getMax(n,y_domain,j)+1;k<=yub[j];k++){
        i = ypl[j*n+k];
        remVal(n,x_domain,i,xPy[i*n+j]);
    }
    yub[j]=getMax(n,y_domain,j);
}

void init(int n, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx, int* xlb){
    for(int i=0;i<n;i++){
        deltaMin(i,n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb);
    }
}