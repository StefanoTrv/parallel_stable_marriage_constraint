#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void print_preference_lists(int n, int xpl[][n], int ypl[][n]) {
    // Men
    printf("\nMen preference lists:\n");
    printf("_\t");
    for (int i = 0; i < n; i++) {
        printf("%d\t", i);
    }
    for (int i = 0; i < n; i++) {
        printf("\n%d:\t", i);
        for (int j = 0; j < n; j++) {
            printf("%d\t", xpl[i][j]);
        }
    }
    
    printf("\n");

    // Women
    printf("\nWomen preference lists:\n");
    printf("_\t");
    for (int i = 0; i < n; i++) {
        printf("%d\t", i);
    }
    for (int i = 0; i < n; i++) {
        printf("\n%d:\t", i);
        for (int j = 0; j < n; j++) {
            printf("%d\t", ypl[i][j]);
        }
    }
            
    printf("\n\n");
}

void print_domains(int n, uint32_t* x_dom, uint32_t* y_dom) {
    int index, offset;
    // Men
    printf("\nMen domains:\n");
    printf("_\t");
    for (int i = 0; i < n; i++) {
        printf("%d\t", i);
    }
    for (int i = 0; i < n; i++) {
        printf("\n%d:\t", i);
        for (int j = 0; j < n; j++) {
            index = i*n+j;
            offset = index % 32;
            printf("%d\t", (x_dom[index/32] << offset) >> (sizeof (int)*8 - 1));
        }
        printf("\n");
    }
    
    printf("\n");

    // Women
    printf("\nWomen domains:\n");
    printf("_\t");
    for (int i = 0; i < n; i++) {
        printf("%d\t", i);
    }
    for (int i = 0; i < n; i++) {
        printf("\n%d:\t", i);
        for (int j = 0; j < n; j++) {
            index = i*n+j;
            offset = index % 32;
            printf("%d\t", (y_dom[index/32] << offset) >> (sizeof (int)*8 - 1));
        }
        printf("\n");
    }
            
    printf("\n\n");
}

void print_reverse_matrixes(int n, int xPy[][n], int yPx[][n]) {
    // Men
    printf("\nMen reversed matrix:\n");
    printf("_\t");
    for (int i = 0; i < n; i++) {
        printf("%d\t", i);
    }
    for (int i = 0; i < n; i++) {
        printf("\n%d:\t", i);
        for (int j = 0; j < n; j++) {
            printf("%d\t", xPy[i][j]);
        }
    }
    
    printf("\n");

    // Women
    printf("\nWomen reversed matrix:\n");
    printf("_\t");
    for (int i = 0; i < n; i++) {
        printf("%d\t", i);
    }
    for (int i = 0; i < n; i++) {
        printf("\n%d:\t", i);
        for (int j = 0; j < n; j++) {
            printf("%d\t", yPx[i][j]);
        }
    }
            
    printf("\n\n");
}

int main() {
    // Example usage
    int n = 3;
    int xpl[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 1, 0}};
    int ypl[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 1, 0}};
    uint32_t* x_dom = malloc(sizeof (uint32_t));
    *x_dom = 2868903936;//{{1, 0, 1}, {0, 1, 0}, {1, 1, 0}};
    uint32_t* y_dom = malloc(sizeof (uint32_t));
    *y_dom = 3925868544;//{{1, 1, 1}, {0, 1, 0}, {0, 1, 0}};
    int xPy[3][3] = {{1, 0, 1}, {0, 1, 0}, {1, 1, 0}};
    int yPx[3][3] = {{1, 0, 1}, {0, 1, 0}, {1, 1, 0}};

    print_preference_lists(n, xpl, ypl);
    print_domains(n, x_dom, y_dom);
    print_reverse_matrixes(n, xPy, yPx);

    return 0;
}