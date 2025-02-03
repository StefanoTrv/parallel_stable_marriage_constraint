#include <stdio.h>
#include "utils/io.c"
#include "utils/printer.c"
#include "domain_functions.c"
#include "utils/error_handler.cu"
#include "cuda_constraint.cu"

void build_reverse_matrix(int,int*, int*);
void get_block_number_and_dimension(int, int, int*, int*);

int main(int argc, char *argv[]) {
    // Get file path from command line arguments or uses a default value
    char *file_path;
    if (argc > 1) {
        file_path = argv[1];
    } else {
        file_path = "input.txt";
    }

    int temp, *temp_p;

    //Reads input data
    int n;
    int *_xpl, *_ypl;
    uint32_t *x_domain, *y_domain;

    parse_input(file_path, &n, &_xpl, &_ypl, &x_domain, &y_domain);

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

    print_preference_lists(n,_xpl,_ypl);

    print_domains(n, x_domain, y_domain);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //Domain device memory allocation and transfer
    uint32_t *d_x_domain, *d_y_domain;
    HANDLE_ERROR(cudaMalloc((void**)&d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMemcpyAsync(d_x_domain, x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(d_y_domain, y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));


    //Host memory allocation
    int length_men_stack, length_women_stack;
    int *xpl = (int *)malloc((n * n * 4 + n * 10 + 2) * sizeof(int));; //only to make the transition to the real constraint easier
    int *ypl = xpl + (n * n);
    memcpy(xpl, _xpl, n * n * sizeof (int));
    memcpy(ypl, _ypl, n * n * sizeof (int));
    int *xPy = ypl + (n * n);
    int *yPx = xPy + (n * n);
    int *stack_mod_men = yPx + (n * n);
    int *stack_mod_women = stack_mod_men + n;
    int *stack_mod_min_men = stack_mod_women + n;
    int *old_min_men = stack_mod_min_men + n;
    int *old_max_men = old_min_men + n;
    int *old_min_women = old_max_men + n;
    int *old_max_women = old_min_women + n;
    int *max_men = old_max_women + n;
    int *min_women = max_men + n;
    int *max_women = min_women + n;
    int *length_min_men_stack = max_women + n;
    int *new_length_min_men_stack = length_min_men_stack + 1;


    //Device memory allocation
    int *d_xpl;
    HANDLE_ERROR(cudaMalloc((void**)&d_xpl, sizeof(int) * (n * n * 4 + n * 12 + 2)));
    int *d_ypl = d_xpl + n * n;
    int *d_xPy = d_ypl + n * n;
    int *d_yPx = d_xPy + n * n;
    int *d_stack_mod_men = d_yPx + n * n;
    int *d_stack_mod_women = d_stack_mod_men + n;
    int *d_stack_mod_min_men = d_stack_mod_women + n;
    int *d_old_min_men = d_stack_mod_min_men + n;
    int *d_old_max_men = d_old_min_men + n;
    int *d_old_min_women = d_old_max_men + n;
    int *d_old_max_women = d_old_min_women + n;
    int *d_max_men = d_old_max_women + n;
    int *d_min_women = d_max_men + n;
    int *d_max_women = d_min_women + n;
    int *d_length_min_men_stack = d_max_women + n;
    int *d_new_length_min_men_stack = d_length_min_men_stack + 1;
    int *d_new_stack_mod_min_men = d_new_length_min_men_stack + 1;
    int *d_array_min_mod_men = d_new_stack_mod_min_men + n;

    //Prepares all the data structures
    build_reverse_matrix(n,xpl,xPy);
    build_reverse_matrix(n,ypl,yPx);

    length_men_stack = n;
    length_women_stack = n;
    *length_min_men_stack = 0; //for f1 we pretend that it's empty, then we fill it before f2
    *new_length_min_men_stack = 0;
    for (int i=0;i<n;i++){
        stack_mod_men[i]=i;
        stack_mod_women[i]=i;
    }

    for(int i=0;i<n;i++){
        old_min_men[i]=0;
        old_min_women[i]=0;
        old_max_men[i]=n-1;
        old_max_women[i]=n-1;
    }

    for(int i=0;i<n;i++){
        //initializes for the case of empty domains
        min_women[i]=n;
        max_men[i]=n-1;
        max_women[i]=n-1;
        temp=n-1;
        while(temp>=0&&getDomainBit(x_domain,i,temp,n)==0){
            //printf("Found empty for man %i value %i",i,temp);
            temp--;
        }
        max_men[i]=temp;
        //printf("max men[%i]=%i\n",i,max_men[i]);
        temp=0;
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

    //Copy into device memory
    HANDLE_ERROR(cudaMemcpyAsync(d_xpl, xpl, (n * n * 4 + n * 10 + 2) * sizeof(int), cudaMemcpyHostToDevice, stream));

    //runs kernels
    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int n_SMP = props.multiProcessorCount;
    int n_threads = length_men_stack + length_women_stack;
    int n_blocks, block_size;
    get_block_number_and_dimension(n_threads,n_SMP,&block_size,&n_blocks);
    
    make_domains_coherent<<<n_blocks,block_size,0,stream>>>(n,d_xpl,d_ypl,d_xPy,d_yPx,d_x_domain,d_y_domain, d_stack_mod_men, d_stack_mod_women, length_men_stack, length_women_stack, d_stack_mod_min_men, d_length_min_men_stack, d_old_min_men, d_old_max_men, d_old_min_women, d_old_max_women);
    cudaStreamSynchronize(stream);

    //debug
    //printf("After f1:\n");
    //HANDLE_ERROR(cudaMemcpy(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    //print_domains(n,x_domain,y_domain);
    //debug

    //empties d_array_min_mod_men
    HANDLE_ERROR(cudaMemsetAsync(d_array_min_mod_men,0,sizeof(int)*n, stream));

    //completely fills min_men_stack
    *length_min_men_stack = n;
    for (int i=0;i<n;i++){
        stack_mod_min_men[i]=i;
    }
    HANDLE_ERROR(cudaMemcpyAsync(d_stack_mod_min_men, stack_mod_min_men, sizeof(int) * n, cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(d_length_min_men_stack, length_min_men_stack, sizeof(int), cudaMemcpyHostToDevice, stream));

    n_threads = *length_min_men_stack;
    get_block_number_and_dimension(n_threads,n_SMP,&block_size,&n_blocks);
    
    //printf("Prima di lancio di f2: %i, %i, %i\n", n_threads, n_blocks,block_size);

    //printf("new_length_min_men_stack vale: %i\n",*new_length_min_men_stack);
    apply_sm_constraint<<<n_blocks,block_size,0,stream>>>(n,d_xpl,d_ypl,d_xPy,d_yPx,d_x_domain,d_y_domain, d_array_min_mod_men, d_stack_mod_min_men, d_length_min_men_stack, d_new_stack_mod_min_men, d_new_length_min_men_stack, d_old_min_men, d_max_men, d_max_women);
    HANDLE_ERROR(cudaMemcpyAsync(new_length_min_men_stack, d_new_length_min_men_stack, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    //printf("new_length_min_men_stack vale: %i\n",*new_length_min_men_stack);
    while(*new_length_min_men_stack!=0){
        //debug
        //printf("After f1:\n");
        //HANDLE_ERROR(cudaMemcpy(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        //HANDLE_ERROR(cudaMemcpy(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        //print_domains(n,x_domain,y_domain);
        //debug

        HANDLE_ERROR(cudaMemsetAsync(d_array_min_mod_men,0,sizeof(int)*n, stream));
        *length_min_men_stack = *new_length_min_men_stack;
        *new_length_min_men_stack = 0;
        HANDLE_ERROR(cudaMemcpyAsync(d_length_min_men_stack, length_min_men_stack, sizeof(int) * 2, cudaMemcpyHostToDevice, stream));
        temp_p = d_new_stack_mod_min_men;
        d_new_stack_mod_min_men = d_stack_mod_min_men;
        d_stack_mod_min_men = temp_p;
        n_threads = *length_min_men_stack;
        get_block_number_and_dimension(n_threads,n_SMP,&block_size,&n_blocks);
        apply_sm_constraint<<<n_blocks,block_size,0,stream>>>(n,d_xpl,d_ypl,d_xPy,d_yPx,d_x_domain,d_y_domain, d_array_min_mod_men, d_stack_mod_min_men, d_length_min_men_stack, d_new_stack_mod_min_men, d_new_length_min_men_stack, d_old_min_men, d_max_men, d_max_women);
        HANDLE_ERROR(cudaMemcpyAsync(new_length_min_men_stack, d_new_length_min_men_stack, sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    //debug
    //printf("After f2:\n");
    //HANDLE_ERROR(cudaMemcpy(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    //print_domains(n,x_domain,y_domain);
    //debug

    n_threads = n;
    get_block_number_and_dimension(n_threads,n_SMP,&block_size,&n_blocks);
    finalize_changes<<<n_blocks,block_size,0,stream>>>(n,d_x_domain,d_y_domain, d_old_min_men, d_old_max_men, d_old_min_women, d_old_max_women, d_max_men, d_min_women, d_max_women);

    //copies from device memory
    HANDLE_ERROR(cudaMemcpyAsync(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaMemcpyAsync(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaMemcpyAsync(old_min_men, d_old_min_men, sizeof(int) * n * 4, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    //frees device memory
    HANDLE_ERROR(cudaFree(d_xpl));
    HANDLE_ERROR(cudaFree(d_x_domain));
    HANDLE_ERROR(cudaFree(d_y_domain));
    cudaStreamDestroy(stream);

    print_domains(n,x_domain,y_domain);

    printf("Men best:\n");
    for(int i = 0;i<n;i++){
        if(getMax(n,y_domain,i)<0){
            printf("EMPTYDOM_for_woman_'%i'",i);
            return 0;
        }
        if(getMin(n,x_domain,i)>n-1){
            printf("EMPTYDOM_for_man_'%i'",i);
        } else{
            printf("%i, ",xpl[i*n+getMin(n,x_domain,i)]);
        }
    }
    printf("\n(in the index domain):\t");
    for(int i = 0;i<n;i++){
        printf("%i, ",getMin(n,x_domain,i));
    }
    printf("\n(old_min_men):\t");
    for(int i = 0;i<n;i++){
        printf("%i, ",old_min_men[i]);
    }
    
    
    //Frees memory and closes
    free(xpl);
    free(_xpl);
    free(_ypl);
    free(x_domain);
    free(y_domain);


    printf("\nClosing...");

    return 0;
}

void build_reverse_matrix(int n,int *zpl, int *zPz){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            zPz[i*n+zpl[i*n+j]]=j;
        }
    }
}

/*
    Computes the appropriate block size and number of blocks based on the number of threads required and the number of SMPs
*/
void get_block_number_and_dimension(int n_threads, int n_SMP, int *block_size, int *n_blocks){
    if (n_threads/n_SMP >= 32){ //at least one warp per SMP
        *n_blocks = n_SMP;
        *block_size = (n_threads + *n_blocks - 1) / *n_blocks;
    } else { //less than one warp per SMP
        *block_size = 32;
        *n_blocks = (n_threads + 31) / 32;
    }
}