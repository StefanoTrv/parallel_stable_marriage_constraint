#include <stdio.h>
#include "utils/io.c"
#include "utils/printer.c"
#include "stable_marriage_constraint.c"

void build_reverse_matrix(int,int*, int*);

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

    //applies once the constraint
    uint32_t *old_x_domain = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    uint32_t *old_y_domain = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    uint32_t *prev_x_domain = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    uint32_t *prev_y_domain = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    for(int i=0;i<(n * n) / 32 + (n % 32 != 0);i++){
        old_x_domain[i]=x_domain[i];
        prev_x_domain[i]=x_domain[i];
        old_y_domain[i]=y_domain[i];
        prev_y_domain[i]=y_domain[i];
    }

    init(n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb);
    int stop;
    while(1){
        stop=1;
        for(int i=0;i<n;i++){
            if(getMin(n,x_domain,i)!=getMin(n,old_x_domain,i)){
                deltaMin(i,n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb);
                printf("deltaMin %i\n",i);
                print_domains(n,x_domain,y_domain);
                stop=0;
            }
            if(getMax(n,y_domain,i)!=getMax(n,old_y_domain,i)){
                deltaMax(i,n,x_domain,y_domain,xpl,ypl,xPy,yPx,yub);
                printf("deltaMax %i\n",i);
                print_domains(n,x_domain,y_domain);
                stop=0;
            }
            for(int k=getMin(n,x_domain,i)+1;k<getMax(n,x_domain,i);k++){
                if(getDomainBit(x_domain,i,k,n)!=getDomainBit(old_x_domain,i,k,n)){
                    removeValue(i,k,n,x_domain,y_domain,xpl,ypl,xPy,yPx);
                    printf("removeValue %i %i\n",i,k);
                    print_domains(n,x_domain,y_domain);
                    stop=0;
                }
            }
        }

        if(stop){
            break;
        }

        printf("I have not stopped!\n");

        //updates old domains
        for(int i=0;i<(n * n) / 32 + (n % 32 != 0);i++){
            old_x_domain[i]=prev_x_domain[i];
            old_y_domain[i]=prev_y_domain[i];
            prev_x_domain[i]=x_domain[i];
            prev_y_domain[i]=y_domain[i];
        }
    }

    print_domains(n,x_domain,y_domain);

    printf("Men best:\n");
    for(int i = 0;i<n;i++){
        printf("%i",xpl[i*n+getMin(n,x_domain,i)]);
    }
    
    
    //Frees memory and closes
    free(xpl);
    free(ypl);
    free(x_domain);
    free(y_domain);
    free(xPy);
    free(yPx);
    free(xlb);
    free(yub);
    free(old_x_domain);
    free(old_y_domain);
    free(prev_x_domain);
    free(prev_y_domain);

    return 0;
}

void build_reverse_matrix(int n,int *zpl, int *zPz){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            zPz[i*n+zpl[i*n+j]]=j;
        }
    }
}