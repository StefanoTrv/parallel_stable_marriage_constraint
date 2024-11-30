/*
    Merges the serial and cuda mockups to test if they give the same results on randomly generated test cases.
*/

#include <stdio.h>
#include "utils/io.c"
#include "utils/printer.c"
#include "stable_marriage_constraint.c"
//#include "domain_functions.c"
#include "utils\error_handler.cu"
#include "cuda_costraint.cu"
#include "utils\pcg-c-basic-0.9\pcg_basic.c"

void build_reverse_matrix(int,int*, int*);
void serial_constraint(int, int*, int*, uint32_t*, uint32_t*);
void cuda_constraint(int, int*, int*, uint32_t*, uint32_t*);
void build_reverse_matrix(int,int*, int*);
int* make_random_preference_matrix(int);
uint32_t* make_full_domain(int);
uint32_t* make_partial_domain(int);

/*
    Executes the tests.
    Use: takes as input the size of the test instances (the value of n), the number of tests to be executed with a complete domain,
    and the number of tests to be executed with randomly filled domains.
*/
int main(int argc, char *argv[]) {
    //get parameters from command line arguments
    int n, compl_tests, incompl_tests;
    if (argc == 4) {
        n = strtol(argv[1],NULL,10);
        compl_tests = strtol(argv[2],NULL,10);
        incompl_tests = strtol(argv[3],NULL,10);
    } else {
        printf("Process interrupted: wrong number of arguments.\nUse:\ncmp_constr <n> <number_of_tests_with_complete_domains> <number_of_tests_with_incomplete_domains>");
        return 5;
    }
    int total_tests = compl_tests + incompl_tests;

    printf("\n%i %i %i\n",n,compl_tests,incompl_tests);

    //Initializes (seeds) global rng (see documentation at https://www.pcg-random.org/using-pcg-c-basic.html#pcg32-srandom-r-rngptr-initstate-initseq)
    pcg32_srandom(42, 42);

    int *men_pl, *women_pl;
    uint32_t *men_domain, *women_domain;

    for(int i=0; i<total_tests; i++){
        men_pl = make_random_preference_matrix(n);
        women_pl = make_random_preference_matrix(n);
        if(i<compl_tests){
            men_domain = make_full_domain(n);
            women_domain = make_full_domain(n);
        } else {
            men_domain = make_partial_domain(n);
            women_domain = make_partial_domain(n);
        }
        printf("Test number %i\n",i);
        print_preference_lists(n,men_pl,women_pl);
        print_domains(n,men_domain,women_domain);

        //TODO copy the domains before passing them
        //serial_constraint(n,men_pl,women_pl,men_domain,women_domain);
        cuda_constraint(n,men_pl,women_pl,men_domain,women_domain);
    }

    //cuda_constraint(0,NULL,NULL,NULL,NULL);

    free(men_pl);
    free(women_pl);
    free(men_domain);
    free(women_domain);

    return 0;
}


/*
    SERIAL CONSTRAINT
*/

void serial_constraint(int n, int *xpl, int *ypl, uint32_t *x_domain, uint32_t *y_domain) {

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
    printf("\n(in the index domain):\t");
    for(int i = 0;i<n;i++){
        printf("%i",getMin(n,x_domain,i));
    }
    
    
    //Frees memory and closes
    free(xPy);
    free(yPx);
    free(xlb);
    free(yub);
    free(old_x_domain);
    free(old_y_domain);
    free(prev_x_domain);
    free(prev_y_domain);
}

/*
    CUDA CONSTRAINT
*/

void cuda_constraint(int n, int *xpl, int *ypl, uint32_t *x_domain, uint32_t *y_domain) {
    int temp, *temp_p;

    //Builds the reverse matrixes
    int *xPy, *yPx;
    xPy = (int *)malloc(n * n * sizeof(int));
    yPx = (int *)malloc(n * n * sizeof(int));
    build_reverse_matrix(n,xpl,xPy);
    build_reverse_matrix(n,ypl,yPx);

    print_reverse_matrixes(n,xPy,yPx);

    //prepares other data and copies it into device memory
    int *d_xpl, *d_ypl, *d_xPy, *d_yPx;
    uint32_t *d_x_domain, *d_y_domain;

    HANDLE_ERROR(cudaMalloc((void**)&d_xpl, sizeof(int) * n * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_ypl, sizeof(int) * n * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_xPy, sizeof(int) * n * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_yPx, sizeof(int) * n * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t)));

    HANDLE_ERROR(cudaMemcpy(d_xpl, xpl, sizeof(int) * n * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_ypl, ypl, sizeof(int) * n * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_xPy, xPy, sizeof(int) * n * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_yPx, yPx, sizeof(int) * n * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_x_domain, x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_y_domain, y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyHostToDevice));

    int *array_mod_men, *array_mod_women, *array_min_mod_men, *stack_mod_men, *stack_mod_women, *stack_mod_min_men, *new_stack_mod_min_men, *length_men_stack, *length_women_stack, *length_min_men_stack, *new_length_min_men_stack;
	HANDLE_ERROR(cudaHostAlloc((void**)&array_mod_men, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&array_mod_women, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&array_min_mod_men, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&stack_mod_men, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&stack_mod_women, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&stack_mod_min_men, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&new_stack_mod_min_men, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&length_men_stack, sizeof (int), cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&length_women_stack, sizeof (int), cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&length_min_men_stack, sizeof (int), cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&new_length_min_men_stack, sizeof (int), cudaHostAllocMapped));
    *length_men_stack = n;
    *length_women_stack = n;
    *length_min_men_stack = n;
    *new_length_min_men_stack = 0;
    for (int i=0;i<n;i++){
        array_mod_men[i]=1;
        array_mod_women[i]=1;
        array_min_mod_men[i]=1;
        stack_mod_men[i]=i;
        stack_mod_women[i]=i;
        stack_mod_min_men[i]=i;
    }
    int *d_array_mod_men, *d_array_mod_women, *d_array_min_mod_men, *d_stack_mod_men, *d_stack_mod_women, *d_stack_mod_min_men, *d_new_stack_mod_min_men, *d_length_men_stack, *d_length_women_stack, *d_length_min_men_stack, *d_new_length_min_men_stack;
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_array_mod_men, array_mod_men, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_array_mod_women, array_mod_women, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_array_min_mod_men, array_min_mod_men, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_stack_mod_men, stack_mod_men, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_stack_mod_women, stack_mod_women, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_stack_mod_min_men, stack_mod_min_men, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_new_stack_mod_min_men, new_stack_mod_min_men, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_length_men_stack, length_men_stack, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_length_women_stack, length_women_stack, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_length_min_men_stack, length_min_men_stack, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&d_new_length_min_men_stack, new_length_min_men_stack, 0));

    int *old_min_men, *old_max_men, *old_min_women, *old_max_women;
    int *d_old_min_men, *d_old_max_men, *d_old_min_women, *d_old_max_women;
    old_min_men = (int*)malloc(sizeof (int) * n);
    old_max_men = (int*)malloc(sizeof (int) * n);
    old_min_women = (int*)malloc(sizeof (int) * n);
    old_max_women = (int*)malloc(sizeof (int) * n);
    for(int i=0;i<n;i++){
        old_min_men[i]=0;
        old_min_women[i]=0;
        old_max_men[i]=n-1;
        old_max_women[i]=n-1;
    }
    HANDLE_ERROR(cudaMalloc((void**)&d_old_min_men, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_old_max_men, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_old_min_women, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_old_max_women, sizeof(int) * n));
    HANDLE_ERROR(cudaMemcpy(d_old_min_men, old_min_men, sizeof(int) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_old_max_men, old_max_men, sizeof(int) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_old_min_women, old_min_women, sizeof(int) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_old_max_women, old_max_women, sizeof(int) * n, cudaMemcpyHostToDevice));
	
    //  computes the vectors of maxes and mins
    //  it may or may not be more efficient to compute after make_domains_coherent, depending on the implementations of the maxes and mins of the domains in the solver
    int *min_men, *max_men, *min_women, *max_women;
    int *d_min_men, *d_max_men, *d_min_women, *d_max_women;
    min_men = (int*)malloc(sizeof (int) * n);
    max_men = (int*)malloc(sizeof (int) * n);
    min_women = (int*)malloc(sizeof (int) * n);
    max_women = (int*)malloc(sizeof (int) * n);
    for(int i=0;i<n;i++){
        //initializes for the case of empty domains
        min_men[i]=n;
        min_women[i]=n;
        max_men[i]=n-1;
        max_women[i]=n-1;
        temp=0;
        while(temp<n&&getDomainBit(x_domain,i,temp,n)==0){
            temp++;
        }
        min_men[i]=temp;
        if(temp==n){//empty domain
            max_men[i]=n-1;
        }else{
            temp=n-1;
            while(getDomainBit(x_domain,i,temp,n)==0){//doesn't need to check for temp>=0 since we know it's not empty
                printf("Found empty for man %i value %i",i,temp);
                temp--;
            }
            max_men[i]=temp;
            printf("max men[%i]=%i\n",i,max_men[i]);
        }
        while(temp<n&&getDomainBit(y_domain,i,temp,n)==0){
            temp++;
        }
        min_women[i]=temp;
        if(temp==n){//empty domain
            max_women[i]=n-1;
        }else{
            temp=n-1;
            while(getDomainBit(y_domain,i,temp,n)==0){//doesn't need to check for temp>=0 since we know it's not empty
                temp--;
            }
            max_women[i]=temp;
        }
    }
    HANDLE_ERROR(cudaMalloc((void**)&d_min_men, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_max_men, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_min_women, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_max_women, sizeof(int) * n));
    HANDLE_ERROR(cudaMemcpy(d_min_men, min_men, sizeof(int) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_max_men, max_men, sizeof(int) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_min_women, min_women, sizeof(int) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_max_women, max_women, sizeof(int) * n, cudaMemcpyHostToDevice));

    //runs kernels
    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int n_blocks = props.multiProcessorCount;
    int block_size = (n + n_blocks - 1) / n_blocks;
    
    make_domains_coherent<<<n_blocks,block_size>>>(n,d_xpl,d_ypl,d_xPy,d_yPx,d_x_domain,d_y_domain,d_array_mod_men, d_array_mod_women, d_array_min_mod_men, d_stack_mod_men, d_stack_mod_women, d_length_men_stack, d_length_women_stack, d_stack_mod_min_men, d_length_min_men_stack, d_old_min_men, d_old_max_men, d_old_min_women, d_old_max_women);

    //debug
    printf("After f1:\n");
    HANDLE_ERROR(cudaMemcpy(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_domains(n,x_domain,y_domain);
    //debug

    //empties array_min_mod_men
    HANDLE_ERROR(cudaMemset(d_array_min_mod_men,0,sizeof(int)*n));

    block_size = (*length_min_men_stack + n_blocks - 1) / n_blocks;
    printf("new_length_min_men_stack vale: %i\n",*new_length_min_men_stack);
    apply_sm_constraint<<<n_blocks,block_size>>>(n,d_xpl,d_ypl,d_xPy,d_yPx,d_x_domain,d_y_domain, d_array_min_mod_men, d_stack_mod_min_men, d_length_min_men_stack, d_new_stack_mod_min_men, d_new_length_min_men_stack, d_old_min_men, d_old_max_men, d_old_min_women, d_old_max_women, d_min_men, d_max_men, d_min_women, d_max_women);
    printf("new_length_min_men_stack vale: %i\n",*new_length_min_men_stack);
    while(*new_length_min_men_stack!=0){
        //debug
        printf("After f1:\n");
        HANDLE_ERROR(cudaMemcpy(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        print_domains(n,x_domain,y_domain);
        //debug

        HANDLE_ERROR(cudaMemset(d_array_min_mod_men,0,sizeof(int)*n));
        *length_min_men_stack = *new_length_min_men_stack;
        *new_length_min_men_stack = 0;
        temp_p = d_new_stack_mod_min_men;
        d_new_stack_mod_min_men = d_stack_mod_min_men;
        d_stack_mod_min_men = temp_p;
        //temp_p = new_stack_mod_min_men;
        //new_stack_mod_min_men = stack_mod_min_men;
        //stack_mod_min_men = temp_p;
        apply_sm_constraint<<<n_blocks,block_size>>>(n,d_xpl,d_ypl,d_xPy,d_yPx,d_x_domain,d_y_domain, d_array_min_mod_men, d_stack_mod_min_men, d_length_min_men_stack, d_new_stack_mod_min_men, d_new_length_min_men_stack, d_old_min_men, d_old_max_men, d_old_min_women, d_old_max_women, d_min_men, d_max_men, d_min_women, d_max_women);
    }

    //debug
    printf("After f2:\n");
    HANDLE_ERROR(cudaMemcpy(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_domains(n,x_domain,y_domain);
    //debug

    block_size = (n + n_blocks - 1) / n_blocks;
    finalize_changes<<<n_blocks,block_size>>>(n,d_xpl,d_ypl,d_xPy,d_yPx,d_x_domain,d_y_domain,d_array_mod_men, d_array_mod_women, d_array_min_mod_men, d_stack_mod_men, d_stack_mod_women, d_length_men_stack, d_length_women_stack, d_stack_mod_min_men, d_length_min_men_stack, d_old_min_men, d_old_max_men, d_old_min_women, d_old_max_women, d_min_men, d_max_men, d_min_women, d_max_women);

    //copies from device memory
    HANDLE_ERROR(cudaMemcpy(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(old_min_men, d_old_min_men, sizeof(int) * n, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(old_max_men, d_old_max_men, sizeof(int) * n, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(old_min_women, d_old_min_women, sizeof(int) * n, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(old_max_women, d_old_max_women, sizeof(int) * n, cudaMemcpyDeviceToHost));

    //sets the lenghts to 0 (useless in this mockup)
    *length_men_stack = 0;
    *length_women_stack = 0;
    *length_min_men_stack = 0;

    //frees device memory
    HANDLE_ERROR(cudaFree(d_xpl));
	HANDLE_ERROR(cudaFree(d_ypl));
	HANDLE_ERROR(cudaFree(d_xPy));
	HANDLE_ERROR(cudaFree(d_yPx));
	HANDLE_ERROR(cudaFree(d_x_domain));
	HANDLE_ERROR(cudaFree(d_y_domain));
    
    HANDLE_ERROR(cudaFreeHost(array_mod_men));
    HANDLE_ERROR(cudaFreeHost(array_mod_women));
    HANDLE_ERROR(cudaFreeHost(array_min_mod_men));
    HANDLE_ERROR(cudaFreeHost(stack_mod_men));
    HANDLE_ERROR(cudaFreeHost(stack_mod_women));
    HANDLE_ERROR(cudaFreeHost(stack_mod_min_men));
    HANDLE_ERROR(cudaFreeHost(new_stack_mod_min_men));
    HANDLE_ERROR(cudaFreeHost(length_men_stack));
    HANDLE_ERROR(cudaFreeHost(length_women_stack));
    HANDLE_ERROR(cudaFreeHost(length_min_men_stack));
    HANDLE_ERROR(cudaFreeHost(new_length_min_men_stack));

    HANDLE_ERROR(cudaFree(d_old_min_men));
    HANDLE_ERROR(cudaFree(d_old_max_men));
    HANDLE_ERROR(cudaFree(d_old_min_women));
    HANDLE_ERROR(cudaFree(d_old_max_women));
	

    print_domains(n,x_domain,y_domain);

    printf("Men best:\n");
    for(int i = 0;i<n;i++){
        if(getMax(n,y_domain,i)<0){
            printf("EMPTYDOM_for_woman_'%i'",i);
            break;
        }
        if(getMin(n,x_domain,i)<n){
            printf("%i",xpl[i*n+getMin(n,x_domain,i)]);
        } else{
            printf("EMPTYDOM_for_man_'%i'",i);
        }
    }
    printf("\n(in the index domain):\t");
    for(int i = 0;i<n;i++){
        printf("%i",getMin(n,x_domain,i));
    }
    
    
    //Frees memory and closes
    free(xPy);
    free(yPx);
    free(old_min_men);
    free(old_min_women);
    free(old_max_men);
    free(old_max_women);
}

/*
    SHARED UTILS
*/

void build_reverse_matrix(int n,int *zpl, int *zPz){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            zPz[i*n+zpl[i*n+j]]=j;
        }
    }
}

/*
    Creates a random n*n preference matrix
    Uses global rng defined in the pcg-basic-c library
*/
int* make_random_preference_matrix(int n){
	int* preference_matrix = (int*)malloc(sizeof (int) * n * n);
	int* numbers = (int*)malloc(sizeof (int) * n);
	int x,t;
	for(int j=0;j<n;j++){
		numbers[j]=j;
	}
	for(int i=0;i<n;i++){
		for(int j=0;j<n-1;j++){
			x = (int) pcg32_boundedrand(n-j);
			preference_matrix[i*n+j]=numbers[x];
            t = numbers[x];
			numbers[x]=numbers[n-j-1];
            //moves the numbers instead of overwriting them, so to not have to initialize the vector everytime
			numbers[n-j-1]=t;
		}
		preference_matrix[i*n+n-1]=numbers[0];
	}
	free(numbers);
	return preference_matrix;
}

/*
    Creates a full domain of given size
*/
uint32_t* make_full_domain(int n){
    uint32_t *domain = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    for(int i=0;i<(n * n) / 32 + (n % 32 != 0);i++){
        domain[i] = 4294967295;
    }
    return domain;
}

/*
    Creates a randomized domain of given size
    Uses global rng defined in the pcg-basic-c library
*/
uint32_t* make_partial_domain(int n){
    uint32_t *domain = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    for(int i=0;i<(n * n) / 32 + (n % 32 != 0);i++){
        domain[i] = pcg32_random();
    }
    return domain;
}