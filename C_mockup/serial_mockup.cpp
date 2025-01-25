#include <stdio.h>
#include "utils/io.c"
#include "utils/printer.c"
#include "serial_constraint.cpp"

#include <queue>


void build_reverse_matrix(int,int*, int*);
void functionDispatcher(std::queue<constraintCall>*, int, uint32_t*, uint32_t*, int*, int*, int*, int*, int*, int*);

int main(int argc, char *argv[]) {
    // Get file path from command line arguments or uses a default value
    char *file_path;
    if (argc > 1) {
        file_path = argv[1];
    } else {
        file_path = "input.txt";
    }

    //Reads input data
    int n;
    int *xpl, *ypl;
    uint32_t *x_domain, *y_domain;

    parse_input(file_path, &n, &xpl, &ypl, &x_domain, &y_domain);

    //If the input didn't include the domains, all the domains are initialized as full
    if(x_domain==NULL){
        x_domain = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
        y_domain = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
        // Read x_domain
        for(int i=0;i<(n * n) / 32 + (n % 32 != 0);i++){
            x_domain[i] = 4294967295;
            y_domain[i] = 4294967295;
        }
    }

    print_preference_lists(n,xpl,ypl);

    print_domains(n, x_domain, y_domain);

    //Builds the reverse matrixes
    int *xPy, *yPx;
    xPy = (int *)malloc(n * n * sizeof(int));
    yPx = (int *)malloc(n * n * sizeof(int));
    build_reverse_matrix(n,xpl,xPy);
    build_reverse_matrix(n,ypl,yPx);

    print_reverse_matrixes(n,xPy,yPx);
    
    //Initializes xlb and yub
    int *xlb = (int *)malloc(n*sizeof(int));
    int *yub = (int *)malloc(n*sizeof(int));
    for(int i=0;i<n;i++){
        xlb[i]=0;
        yub[i]=n-1;
    }

    std::queue<constraintCall> queue;

    //Looks for missing values in the domains and adds the proper function to the queue
    for(int i=0;i<n;i++){
        if(getMin(n,x_domain,i)>=n||getMax(n,y_domain,i)<0){
            printf("\n-------------------\nFound empty domain!\n-------------------\n");
            print_domains(n,x_domain,y_domain);
            return -1;
        }
        if(getMin(n,x_domain,i)!=0){
            queue.push(constraintCall(1,i,0,0));
        }
        if(getMax(n,y_domain,i)!=n-1){
            queue.push(constraintCall(2,i,0,0));
        }
        for(int k=getMin(n,x_domain,i)+1;k<n;k++){
            if(getDomainBit(x_domain,i,k,n)!=1){
                queue.push(constraintCall(0,i,k,1));
            }
        }
        //Applies remove value on the women too (this is missing from the original paper)
        for(int k=0;k<getMax(n,y_domain,i);k++){
            if(getDomainBit(y_domain,i,k,n)!=1){
                queue.push(constraintCall(0,i,k,0));
            }
        }
    }

    //Runs init
    init(n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb,&queue);

    // Executes functions until the queue is empty
    while(!queue.empty()){
        functionDispatcher(&queue,n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb,yub);
    }
    
    // Checks if there's an empty domain
    for(int i=0;i<n;i++){
        //printf("%iith iteration\n",i);
        if(getMin(n,x_domain,i)>=n||getMax(n,y_domain,i)<0){
            printf("\n-------------------\nFound empty domain!\n-------------------\n");
            print_domains(n,x_domain,y_domain);
            return -1;
        }
    }

    print_domains(n,x_domain,y_domain);

    printf("Men best:\n");
    for(int i = 0;i<n;i++){
        printf("%i ",xpl[i*n+getMin(n,x_domain,i)]);
    }
    printf("\n(in the index domain):\t");
    for(int i = 0;i<n;i++){
        printf("%i ",getMin(n,x_domain,i));
    }
    printf("\n");
    
    
    //Frees memory and closes
    free(xpl);
    free(ypl);
    free(x_domain);
    free(y_domain);
    free(xPy);
    free(yPx);
    free(xlb);
    free(yub);

    return 0;
}

void functionDispatcher(std::queue<constraintCall> *queue, int n, uint32_t* x_domain, uint32_t* y_domain, int* xpl, int* ypl, int* xPy, int* yPx, int* xlb, int* yub){
    constraintCall c = queue->front();
    switch (c.function) {
        case 0: // removeValue
            removeValue(c.ij,c.a,c.isMan,n,x_domain,y_domain,xpl,ypl,xPy,yPx,queue);
            break;

        case 1: // deltaMin
            deltaMin(c.ij,n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb,queue);
            break;

        case 2: // deltaMax
            deltaMax(c.ij,n,x_domain,y_domain,xpl,ypl,xPy,yPx,yub,queue);
            break;
    }
    queue->pop();
}

void build_reverse_matrix(int n,int *zpl, int *zPz){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            zPz[i*n+zpl[i*n+j]]=j;
        }
    }
}