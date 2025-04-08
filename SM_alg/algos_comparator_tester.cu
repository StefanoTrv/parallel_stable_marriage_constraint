/*
    Merges the serial and cuda mockups to test if they give the same results on randomly generated test cases.
*/

#include <stdio.h>
#include "utils/printer.c"
#include "utils/error_handler.cu"
#include "cuda_SM.cu"
#include "utils/pcg-c-basic-0.9/pcg_basic.c"
#include <queue>
#include <chrono>

void build_reverse_matrix(int,int*, int*);
void cuda_stable_marriage(int, int*, int*, int*);
void build_reverse_matrix(int,int*, int*);
int* make_random_preference_matrix(int, int*);
void make_full_domain(int,uint32_t*);
void get_block_number_and_dimension(int, int, int*, int*);
void extended_gale_shapley(int, int*, int*, int*);
void print_results(int, int*, int*);
void clone_pl(int, int*, int*);

/*
    Executes the tests.
    Use: takes as input the size of the test instances (the value of n), the number of tests to be executed with a complete domain,
    and the number of tests to be executed with randomly filled domains.
*/
int main(int argc, char *argv[]) {
    //get parameters from command line arguments
    int n, n_tests;
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::microseconds duration_cuda, duration_serial;

    if (argc == 3) {
        n = strtol(argv[1],NULL,10);
        n_tests = strtol(argv[2],NULL,10);
    } else {
        printf("Process interrupted: wrong number of arguments.\nUse:\ncmp_algs <n> <number_of_tests>");
        return 5;
    }

    //Initializes (seeds) global rng (see documentation at https://www.pcg-random.org/using-pcg-c-basic.html#pcg32-srandom-r-rngptr-initstate-initseq)
    pcg32_srandom(42, 42);

	int* mpl = (int*)malloc(sizeof (int) * n * n);
	int* wpl = (int*)malloc(sizeof (int) * n * n);
	int* mpl_cpy = (int*)malloc(sizeof (int) * n * n);
	int* wpl_cpy = (int*)malloc(sizeof (int) * n * n);

    int *results_egs = (int*) malloc(n * sizeof (int));
    int *results_cuda = (int*) malloc(n * sizeof (int));

    for(int i=0; i<n_tests; i++){
        mpl = make_random_preference_matrix(n,mpl);
        wpl = make_random_preference_matrix(n,wpl);
        clone_pl(n,mpl,mpl_cpy);
        clone_pl(n,wpl,wpl_cpy);

        start = std::chrono::high_resolution_clock::now();
        cuda_stable_marriage(n,mpl,wpl,results_cuda);
        end = std::chrono::high_resolution_clock::now();

        duration_cuda = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        start = std::chrono::high_resolution_clock::now();
        extended_gale_shapley(n,mpl_cpy,wpl_cpy,results_egs);
        end = std::chrono::high_resolution_clock::now();

        duration_serial = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        printf("%i - %i\n",duration_serial,duration_cuda);

        for(int j=0; j<n; j++){
            if(results_cuda[j]!=results_egs[j]){
                printf("Found error!\n");
                print_results(n,mpl,results_egs);
                print_results(n,mpl,results_cuda);
            }
        }
    }
    free(results_egs);
    free(results_cuda);
    free(mpl);
    free(wpl);
    free(mpl_cpy);
    free(wpl_cpy);

    printf("Done!\n");
    return 0;
}

void cuda_stable_marriage(int n, int *xpl, int *ypl, int* results) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //Domains
    int domain_size = ((n * n) / 32 + (n % 32 != 0));
    uint32_t *x_domain = (uint32_t *)malloc(domain_size * sizeof(uint32_t) * 2);
    uint32_t *y_domain = x_domain + domain_size;
    make_full_domain(n,x_domain);
    make_full_domain(n,y_domain);
    uint32_t *d_x_domain, *d_y_domain;
    HANDLE_ERROR(cudaMalloc((void**)&d_x_domain, domain_size * sizeof(uint32_t) * 2));
    d_y_domain = d_x_domain + domain_size;
    HANDLE_ERROR(cudaMemcpyAsync(d_x_domain, x_domain, domain_size * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    //Host memory allocation
    int *xPy = (int *)malloc((n * n * 2 + n * 2) * sizeof(int));
    int *yPx = xPy + (n * n);
    int *old_min_men = yPx + n * n;
    int *max_women = old_min_men + n;

    //Device memory allocation
    int *d_xpl;
    HANDLE_ERROR(cudaMalloc((void**)&d_xpl, sizeof(int) * (n * n * 4 + n * 2)));
    int *d_ypl = d_xpl + n * n;
    int *d_xPy = d_ypl + n * n;
    int *d_yPx = d_xPy + n * n;
    int *d_old_min_men = d_yPx + n * n;
    int *d_max_women = d_old_min_men + n;

    //copies preference lists into device memory
    HANDLE_ERROR(cudaMemcpyAsync(d_xpl, xpl, (n * n) * sizeof(int), cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(d_ypl, ypl, (n * n) * sizeof(int), cudaMemcpyHostToDevice, stream));

    //Prepares the data structures
    build_reverse_matrix(n,xpl,xPy);
    build_reverse_matrix(n,ypl,yPx);

    for(int i=0;i<n;i++){
        old_min_men[i]=0;
        max_women[i]=n-1;
    }

    //Copy other data structures into device memory
    HANDLE_ERROR(cudaMemcpyAsync(d_xPy, xPy, (n * n * 2 + n * 2) * sizeof(int), cudaMemcpyHostToDevice, stream));

    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int n_SMP = props.multiProcessorCount;
    int n_blocks, block_size;
    get_block_number_and_dimension(n,n_SMP,&block_size,&n_blocks);
    
    cuda_stable_marriage_kernel<<<n_blocks,block_size,0,stream>>>(n,d_xpl,d_ypl,d_xPy,d_yPx,d_x_domain,d_y_domain, d_old_min_men, d_max_women);

    //copies from device memory
    HANDLE_ERROR(cudaMemcpyAsync(results, d_old_min_men, sizeof(int) * n, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    //frees device memory
    HANDLE_ERROR(cudaFree(d_xpl));
    HANDLE_ERROR(cudaFree(d_x_domain));
    cudaStreamDestroy(stream);

    //Frees memory and closes
    free(xPy);
    free(x_domain);
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
int* make_random_preference_matrix(int n, int* preference_matrix){
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
    Takes a pointer to a properly allocated domain
*/
void make_full_domain(int n, uint32_t *domain){
    for(int i=0;i<(n * n) / 32 + (n % 32 != 0);i++){
        domain[i] = 4294967295;
    }
}

/*
    Copies a preference list
*/
void clone_pl(int n, int* old_pl, int* new_pl){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            new_pl[i*n+j]=old_pl[i*n+j];
        }
    }
}

/*
    Prints results
*/
void print_results(int n, int* mpl, int* results){
    for(int i=0; i<n-1; i++){
        printf("%i - ",mpl[i*n+results[i]]);
    }
    printf("%i (",mpl[(n-1)*n+results[n-1]]);
    for(int i=0; i<n-1; i++){
        printf("%i - ",results[i]);
    }
    printf("%i)\n",results[n-1]);
}

void extended_gale_shapley(int n, int *mpl, int *wpl, int *m_partners){
    std::queue<int> queue;
    int *men_indexes = m_partners;
    int *women_engagements = (int*) malloc(n * sizeof (int));
    int *women_maxes = (int*) malloc(n * sizeof (int));
    int *mPw = (int*) malloc(n * n * sizeof (int));
    for(int i=0;i<n;i++){
        queue.push(i);
        men_indexes[i] = 0;
        women_engagements[i] = -1;
        women_maxes[i] = n-1;
        for(int j=0;j<n;j++){
            mPw[i*n+mpl[i*n+j]]=j;
        }
    }
    int m, w, p;
    while(!queue.empty()){
        m = queue.front();
        while(mpl[m*n+men_indexes[m]]==-1){ //skips removed values
            men_indexes[m]++;
        }
        w = mpl[m*n+men_indexes[m]];
        if(women_engagements[w]!=-1){
            p = women_engagements[w];
            queue.push(p);
            men_indexes[p]++;
        }
        women_engagements[w] = m;
        p = wpl[w*n+women_maxes[w]];
        while(p != m){
            mpl[p*n+mPw[p*n+w]]=-1;
            women_maxes[w]--;
            p = wpl[w*n+women_maxes[w]];
        }
        queue.pop();
    }
    free(women_engagements);
    free(women_maxes);
    free(mPw);
}
