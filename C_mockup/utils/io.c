#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

const uint32_t UNSIGNED_ONE = 1;

void parse_input(const char *file_path, int *n, int ***xpl, int ***ypl, uint32_t **x_domain, uint32_t **y_domain) {
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        exit(1);
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    int index, offset, temp;
    fseek(file, 0, SEEK_SET);

    // Read n
    fscanf(file, "%d", n);
    *xpl = (int **)malloc((*n) * sizeof(int *));
    *ypl = (int **)malloc((*n) * sizeof(int *));
    *x_domain = NULL;
    *y_domain = NULL;

    // Skip empty line
    fgetc(file);

    // Read xpl
    for (int i = 0; i < *n; i++) {
        (*xpl)[i] = (int *)malloc((*n) * sizeof(int));
        for (int j = 0; j < *n; j++) {
            fscanf(file, "%d", &(*xpl)[i][j]);
        }
    }

    // Skip empty line
    fgetc(file);

    // Read ypl
    for (int i = 0; i < *n; i++) {
        (*ypl)[i] = (int *)malloc((*n) * sizeof(int));
        for (int j = 0; j < *n; j++) {
            fscanf(file, "%d", &(*ypl)[i][j]);
        }
    }

    // Skip empty line
    fgetc(file);

    // Check if domains are present in the file
    if ((file_size - ftell(file)) >= ((*n) * (*n) * 2) / 32) {
        *x_domain = (uint32_t *)malloc((((*n) * (*n)) / 32 + ((*n) % 32 != 0)) * sizeof(uint32_t));
        int *x_dom = (uint32_t *)malloc((((*n) * (*n)) / 32 + ((*n) % 32 != 0)) * sizeof(uint32_t));
        *y_domain = (uint32_t *)malloc((((*n) * (*n)) / 32 + ((*n) % 32 != 0)) * sizeof(uint32_t));
        // Read x_domain
        for (int i = 0; i < *n; i++) {
            for (int j = 0; j < *n; j++) {
                fscanf(file, "%d", &temp);
                index = (i*(*n)+j);
                offset = index % 32;
                if(offset==0){
                    (*x_domain)[index/32] = 0;
                }
                if (temp==0){
                    //bitwise and not
                    (*x_domain)[index/32] = (*x_domain)[index/32] & ~((UNSIGNED_ONE<< (sizeof (int)*8 - 1)) >> offset);
                }else{
                    //bitwise or
                    (*x_domain)[index/32] = (*x_domain)[index/32] | ((UNSIGNED_ONE<< (sizeof (int)*8 - 1)) >> offset);
                }
            }
        }

        // Skip empty line
        fgetc(file);

        // Read y_domain
        for (int i = 0; i < *n; i++) {
            for (int j = 0; j < *n; j++) {
                fscanf(file, "%d", &temp);
                index = (i*(*n)+j);
                offset = index % 32;
                if(offset==0){
                    (*y_domain)[index/32] = 0;
                }
                if (temp==0){
                    //bitwise and not
                    (*y_domain)[index/32] = (*y_domain)[index/32] & ~((UNSIGNED_ONE<< (sizeof (int)*8 - 1)) >> offset);
                }else{
                    //bitwise or
                    (*y_domain)[index/32] = (*y_domain)[index/32] | ((UNSIGNED_ONE<< (sizeof (int)*8 - 1)) >> offset);
                }
            }
        }
    }

    fclose(file);
}

int main() {
    int n, index, offset;
    int **xpl, **ypl;
    uint32_t *x_domain, *y_domain;

    parse_input("input.txt", &n, &xpl, &ypl, &x_domain, &y_domain);

    // Test printing
    printf("n: %d\n", n);
    printf("xpl:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", xpl[i][j]);
        }
        printf("\n");
    }
    printf("ypl:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", ypl[i][j]);
        }
        printf("\n");
    }
    if (x_domain != NULL && y_domain != NULL) {
        printf("x_domain:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                index = i*n+j;
                offset = index % 32;
                printf("%d ", (x_domain[index/32] << offset) >> (sizeof (int)*8 - 1));
            }
            printf("\n");
        }
        printf("y_domain:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                index = i*n+j;
                offset = index % 32;
                printf("%d ", (y_domain[index/32] << offset) >> (sizeof (int)*8 - 1));
            }
            printf("\n");
        }
    }

    // Free memory
    for (int i = 0; i < n; i++) {
        free(xpl[i]);
        free(ypl[i]);
    }
    free(xpl);
    free(ypl);
    free(x_domain);
    free(y_domain);

    return 0;
}
