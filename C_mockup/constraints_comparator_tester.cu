/*
    Merges the serial and cuda mockups to test if they give the same results on randomly generated test cases.
*/

#include <stdio.h>
#include "utils/io.c"
#include "utils/printer.c"
#include "serial_constraint.cpp"
#include "domain_functions.c"
#include "utils/error_handler.cu"
#include "cuda_constraint.cu"
#include "utils/pcg-c-basic-0.9/pcg_basic.c"
#include <queue>

void build_reverse_matrix(int,int*, int*);
int serial_constraint(int, int*, int*, uint32_t*, uint32_t*);
int cuda_constraint(int, int*, int*, uint32_t*, uint32_t*);
void build_reverse_matrix(int,int*, int*);
int* make_random_preference_matrix(int);
void make_full_domain(int,uint32_t*);
void make_partial_domain(int,uint32_t*);
void clone_domain(int, uint32_t*,uint32_t*);
void get_block_number_and_dimension(int, int, int*, int*);
int compare_domains(int, uint32_t*, uint32_t*);
void functionDispatcher(std::queue<constraintCall>*, int, uint32_t*, uint32_t*, int*, int*, int*, int*, int*, int*);
void freeSerialMemory(int*, int*, int*, int*);

/*
    Executes the tests.
    Use: takes as input the size of the test instances (the value of n), the number of tests to be executed with a complete domain,
    and the number of tests to be executed with randomly filled domains.
*/
int main(int argc, char *argv[]) {
    //get parameters from command line arguments
    int n, compl_tests, incompl_tests;
    int empty_domains_founds = 0;
    int errors = 0;
    int empty_notempty_errors = 0;
    int serial_status, cuda_status;
    if (argc == 4) {
        n = strtol(argv[1],NULL,10);
        compl_tests = strtol(argv[2],NULL,10);
        incompl_tests = strtol(argv[3],NULL,10);
    } else {
        printf("Process interrupted: wrong number of arguments.\nUse:\ncmp_constr <n> <number_of_tests_with_complete_domains> <number_of_tests_with_incomplete_domains>\n");
        return 5;
    }
    int total_tests = compl_tests + incompl_tests;

    //Initializes (seeds) global rng (see documentation at https://www.pcg-random.org/using-pcg-c-basic.html#pcg32-srandom-r-rngptr-initstate-initseq)
    pcg32_srandom(42, 42);

    int *men_pl, *women_pl;
    uint32_t *men_domain = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    uint32_t *women_domain = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    uint32_t *men_domain_parallel = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    uint32_t *women_domain_parallel = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    uint32_t *men_domain_orig = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));
    uint32_t *women_domain_orig = (uint32_t *)malloc(((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t));

    for(int i=0; i<total_tests; i++){
        //printf("Beginning test %i\n",i);
        men_pl = make_random_preference_matrix(n);
        women_pl = make_random_preference_matrix(n);
        if(i<compl_tests){
            make_full_domain(n,men_domain);
            make_full_domain(n,women_domain);
        } else {
            make_partial_domain(n,men_domain);
            make_partial_domain(n,women_domain);
        }

        clone_domain(n,men_domain,men_domain_parallel);
        clone_domain(n,women_domain,women_domain_parallel);
        clone_domain(n,men_domain,men_domain_orig);
        clone_domain(n,women_domain,women_domain_orig);

        serial_status = serial_constraint(n,men_pl,women_pl,men_domain,women_domain);
        cuda_status = cuda_constraint(n,men_pl,women_pl,men_domain_parallel,women_domain_parallel);
        
        if (serial_status == -1 && serial_status == cuda_status){ //correct
            empty_domains_founds++;
        } else if(serial_status == cuda_status){ //must check
            if(!compare_domains(n,men_domain,men_domain_parallel) || !compare_domains(n,women_domain,women_domain_parallel)){//not equal
                errors++;
                printf("--------------------------------\n  Found error in test number %i (different domains) \n--------------------------------\n",i);
                print_preference_lists(n,men_pl,women_pl);
                print_domains(n,men_domain_orig,women_domain_orig);
                printf("Resulting domains for serial constraint:\n");
                print_domains(n,men_domain,women_domain);
                printf("Resulting domains for parallel constraint:\n");
                print_domains(n,men_domain_parallel,women_domain_parallel);
            }
        }else{ //surely an error
            errors++;
            empty_notempty_errors++;
            printf("--------------------------------\n  Found error in test number %i (empty and non empty domains) \n--------------------------------\n",i);
            if(serial_status == -1){
                printf("Domain for serial constraint was found to be empty.\n");
            } else {
                printf("Domain for parallel constraint was found to be empty.\n");
            }
            print_preference_lists(n,men_pl,women_pl);
            print_domains(n,men_domain_orig,women_domain_orig);
            printf("Resulting domains for serial constraint:\n");
            print_domains(n,men_domain,women_domain);
            printf("Resulting domains for parallel constraint:\n");
            print_domains(n,men_domain_parallel,women_domain_parallel);
        }
        free(men_pl);
        free(women_pl);
    }
    free(men_domain);
    free(women_domain);
    free(men_domain_parallel);
    free(women_domain_parallel);
    free(men_domain_orig);
    free(women_domain_orig);

    printf("\nTesting complete\n%i errors were found out of %i tests (of which, %i where one domain was empty and the other was not)\n%i empty domains were correctly identified\n",errors,total_tests,empty_notempty_errors,empty_domains_founds);
    return 0;
}


/*
    SERIAL CONSTRAINT
*/

int serial_constraint(int n, int *xpl, int *ypl, uint32_t *x_domain, uint32_t *y_domain) {

    //Returns if it finds an empty domain
    int emptyM, emptyW;
    for(int i=0;i<n;i++){
        emptyM = true;
        emptyW = true;
        for(int j=0;j<n;j++){
            if(getDomainBit(x_domain,i,j,n)){
                emptyM=false;
            }
            if(getDomainBit(y_domain,i,j,n)){
                emptyW=false;
            }
        }
        if(emptyM || emptyW){
            //printf("Found empty domain as input: returning.\n");
            return -1;
        }
    }

    //Builds the reverse matrixes
    int *xPy, *yPx;
    xPy = (int *)malloc(n * n * sizeof(int));
    yPx = (int *)malloc(n * n * sizeof(int));
    build_reverse_matrix(n,xpl,xPy);
    build_reverse_matrix(n,ypl,yPx);

    //print_reverse_matrixes(n,xPy,yPx);

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
            freeSerialMemory(xPy, yPx, xlb, yub);
            return -1;
        }
        if(getVal(n,x_domain,i)!=-1){ //man is bound
            queue.push(constraintCall(3,i,0,1));
        } else {
            if(getMin(n,x_domain,i)!=0){
                queue.push(constraintCall(1,i,0,0));
            }
            for(int k=getMin(n,x_domain,i)+1;k<n;k++){
                if(getDomainBit(x_domain,i,k,n)!=1){
                    queue.push(constraintCall(0,i,k,1));
                }
            }
        }
        if(getVal(n,y_domain,i)!=-1){ //woman is bound
            queue.push(constraintCall(3,i,0,0));
        } else {
            if(getMax(n,y_domain,i)!=n-1){
                queue.push(constraintCall(2,i,0,0));
            }
            //Applies remove value on the women too (this is missing from the original paper)
            for(int k=0;k<getMax(n,y_domain,i);k++){
                if(getDomainBit(y_domain,i,k,n)!=1){
                    queue.push(constraintCall(0,i,k,0));
                }
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
            //printf("\n-------------------\nFound empty domain!\n-------------------\n");
            //print_domains(n,x_domain,y_domain);
            freeSerialMemory(xPy, yPx, xlb, yub);
            return -1;
        }
    }

    //Frees memory and closes
    freeSerialMemory(xPy, yPx, xlb, yub);
    
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
        
        case 3: // inst
            inst(c.ij,n,c.isMan,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb,yub,queue);
            break;
    }
    queue->pop();
}

void freeSerialMemory(int *xPy, int *yPx, int *xlb, int *yub) {
    free(xPy);
    free(yPx);
    free(xlb);
    free(yub);
}

/*
    CUDA CONSTRAINT
*/

int cuda_constraint(int n, int *_xpl, int *_ypl, uint32_t *x_domain, uint32_t *y_domain) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int temp;

    //Domain device memory allocation and transfer
    uint32_t *d_x_domain, *d_y_domain;
    HANDLE_ERROR(cudaMalloc((void**)&d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMemcpyAsync(d_x_domain, x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(d_y_domain, y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));


    //Host memory allocation
    int length_men_stack, length_women_stack;
    int *xpl = (int *)malloc((n * n * 4 + n * 10 + 2) * sizeof(int)); //only to make the transition to the real constraint easier
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
    HANDLE_ERROR(cudaMalloc((void**)&d_xpl, sizeof(int) * (n * n * 4 + n * 12 + 3)));
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
    int *d_warp_counter = d_array_min_mod_men + n;

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
    cudaMemcpyToSymbol(d_n_SMP, &n_SMP, sizeof(int));
    int n_threads = length_men_stack + length_women_stack;
    int n_blocks, block_size;
    get_block_number_and_dimension(n_threads,n_SMP,&block_size,&n_blocks);

    //empties d_array_min_mod_men
    HANDLE_ERROR(cudaMemsetAsync(d_array_min_mod_men,0,sizeof(int)*(n+1), stream));
    
    make_domains_coherent<<<n_blocks,block_size,0,stream>>>(true, n, d_xpl, d_ypl, d_xPy, d_yPx, d_x_domain, d_y_domain, d_array_min_mod_men, d_new_stack_mod_min_men, d_new_length_min_men_stack, d_stack_mod_men, d_stack_mod_women, length_men_stack, length_women_stack, d_stack_mod_min_men, d_length_min_men_stack, d_old_min_men, d_old_max_men, d_old_min_women, d_old_max_women, d_max_men, d_min_women, d_max_women, d_warp_counter);

    //copies from device memory
    HANDLE_ERROR(cudaMemcpyAsync(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaMemcpyAsync(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaMemcpyAsync(old_min_men, d_old_min_men, sizeof(int) * n * 4, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    //debug
    //printf("After kernels:\n");
    //HANDLE_ERROR(cudaMemcpy(x_domain, d_x_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(y_domain, d_y_domain, ((n * n) / 32 + (n % 32 != 0)) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    //print_domains(n,x_domain,y_domain);
    //debug

    //frees device memory
    HANDLE_ERROR(cudaFree(d_xpl));
    HANDLE_ERROR(cudaFree(d_x_domain));
    HANDLE_ERROR(cudaFree(d_y_domain));
    cudaStreamDestroy(stream);

    //checks if there's an empty domain
    int emptyX, emptyY;
    for(int i=0;i<n;i++){
        emptyX = true;
        emptyY = true;
        for(int j=0;j<n;j++){
            if(getDomainBit(x_domain,i,j,n)){
                emptyX=false;
            }
            if(getDomainBit(y_domain,i,j,n)){
                emptyY=false;
            }
        }
        if(emptyX || emptyY){//Frees memory and closes
            //printf("Empty.\n");
            free(xpl);
            return -1;
        }
    }
    
    for(int i=0;i<n;i++){
        if(getMin(n,x_domain,i)<n&&getMin(n,x_domain,i)!=old_min_men[i]){
            printf("Mistake in man %i! Min is %i and old_min is %i!\n",i,getMin(n,x_domain,i),old_min_men[i]);
        }
    }
    for(int i=0;i<n;i++){
        if(getMax(n,x_domain,i)>=0&&getMax(n,x_domain,i)!=old_max_men[i]){
            printf("Mistake in man %i! Max is %i and old_max is %i!\n",i,getMax(n,x_domain,i),old_max_men[i]);
        }
    }
    
    for(int i=0;i<n;i++){
        if(getMin(n,y_domain,i)<n&&getMin(n,y_domain,i)!=old_min_women[i]){
            printf("Mistake in woman %i! Min is %i and old_min is %i!\n",i,getMin(n,y_domain,i),old_min_women[i]);
        }
    }
    for(int i=0;i<n;i++){
        if(getMax(n,y_domain,i)>=0&&getMax(n,y_domain,i)!=old_max_women[i]){
            printf("Mistake in woman %i! Max is %i and old_max is %i!\n",i,getMax(n,y_domain,i),old_max_women[i]);
        }
    }

    //Frees memory and closes
    free(xpl);
    return 0;
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
    Takes a pointer to a properly allocated domain
*/
void make_full_domain(int n, uint32_t *domain){
    for(int i=0;i<(n * n) / 32 + (n % 32 != 0);i++){
        domain[i] = 4294967295;
    }
}

/*
    Creates a randomized domain of given size
    Uses global rng defined in the pcg-basic-c library
    Takes a pointer to a properly allocated domain
*/
void make_partial_domain(int n, uint32_t *domain){
    for(int i=0;i<(n * n) / 32 + (n % 32 != 0);i++){
        domain[i] = pcg32_random() | pcg32_random() | pcg32_random() | pcg32_random() | pcg32_random() | pcg32_random() | pcg32_random() | pcg32_random() | pcg32_random() | pcg32_random();
    }
}

/*
    Taken a domain, it creates a copy
*/
void clone_domain(int n, uint32_t *old_domain, uint32_t *new_domain){
    for(int i=0;i<(n * n) / 32 + (n % 32 != 0);i++){
        new_domain[i] = old_domain[i];
    }
}

/*
    Compares two domains
    For performance reasons, it includes in the comparison the ending part of the allocated memory that doesn't contain any info.
    The correctness of this function depends on how the domains were created.
*/
int compare_domains(int n, uint32_t *d1, uint32_t *d2){
    for(int i=0;i<(n * n) / 32 + (n % 32 != 0);i++){
        if(d1[i]!=d2[i]){
            return 0;
        }
    }
    return 1;
}