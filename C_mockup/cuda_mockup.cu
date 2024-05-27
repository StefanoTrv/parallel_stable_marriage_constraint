#include <stdio.h>
#include "utils/io.c"
#include "utils/printer.c"
#include "domain_functions.c"
#include "utils\error_handler.cu"
#include "cuda_costraint.cu"

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

    //prepares other data and copies into device memory
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

    int *array_mod_men, *array_mod_women, *array_min_mod_men, *stack_mod_men, *stack_mod_women, *stack_mod_min_men;
    int length_men_stack, length_women_stack, length_min_men_stack;
    length_men_stack = n;
    length_women_stack = n;
    length_min_men_stack = n;
	HANDLE_ERROR(cudaHostAlloc((void**)&array_mod_men, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&array_mod_women, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&array_min_mod_men, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&stack_mod_men, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&stack_mod_women, sizeof (int) * n, cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&stack_mod_min_men, sizeof (int) * n, cudaHostAllocMapped));
    for (int i=0;i<n;i++){
        array_mod_men[i]=1;
        array_mod_women[i]=1;
        array_min_mod_men[i]=1;
        stack_mod_men[i]=i;
        stack_mod_women[i]=i;
        stack_mod_min_men[i]=i;
    }

    int *old_min_man, *old_max_man, *old_min_woman, *old_max_woman;
    int *d_old_min_man, *d_old_max_man, *d_old_min_woman, *d_old_max_woman;
    old_min_man = (int*)malloc(sizeof (int) * n);
    old_max_man = (int*)malloc(sizeof (int) * n);
    old_min_woman = (int*)malloc(sizeof (int) * n);
    old_max_woman = (int*)malloc(sizeof (int) * n);
    for(int i=0;i<n;i++){
        old_min_man[i]=0;
        old_min_woman[i]=0;
        old_max_man[i]=n-1;
        old_max_woman[i]=n-1;
    }
    HANDLE_ERROR(cudaMalloc((void**)&d_old_min_man, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_old_max_man, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_old_min_woman, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d_old_max_woman, sizeof(int) * n));
    HANDLE_ERROR(cudaMemcpy(d_old_min_man, old_min_man, sizeof(int) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_old_max_man, old_max_man, sizeof(int) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_old_min_woman, old_min_woman, sizeof(int) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_old_max_woman, old_max_woman, sizeof(int) * n, cudaMemcpyHostToDevice));
	

    //runs kernels
    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int n_blocks = props.multiProcessorCount;
    int block_size = (n + n_blocks - 1) / n_blocks;
    
    my_kernel<<<n_blocks,block_size>>>(n,d_xpl,d_ypl,d_xPy,d_yPx,d_x_domain,d_y_domain);

    //copies from device memory
    HANDLE_ERROR(cudaMemcpy(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

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

    HANDLE_ERROR(cudaFree(d_old_min_man));
    HANDLE_ERROR(cudaFree(d_old_max_man));
    HANDLE_ERROR(cudaFree(d_old_min_woman));
    HANDLE_ERROR(cudaFree(d_old_max_woman));
	

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
    free(old_min_man);
    free(old_min_woman);
    free(old_max_man);
    free(old_max_woman);

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