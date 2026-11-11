#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

/* The assignment specifies a 20 x 20 matrix and a pool of ten threads. */
#define MAX 20
#define NUM_THREADS 10

static int matrixSize = MAX;
static int matA[MAX][MAX];
static int matB[MAX][MAX];
static int matSumResult[MAX][MAX];
static int matDiffResult[MAX][MAX];
static int matProductResult[MAX][MAX];

/* Fill only the active dimension of the fixed-size matrix. */
void fillMatrix(int matrix[MAX][MAX])
{
    for (int i = 0; i < matrixSize; ++i)
        for (int j = 0; j < matrixSize; ++j)
            matrix[i][j] = rand() % 10 + 1;
}

void printMatrix(int matrix[MAX][MAX])
{
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j)
            printf("%5d", matrix[i][j]);
        printf("\n");
    }
    printf("\n");
}

/* Disjoint rows assigned in strides of NUM_THREADS eliminate data races. */
void *computeSum(void *args)
{
    const int id = *(const int *)args;
    for (int row = id; row < matrixSize; row += NUM_THREADS)
        for (int col = 0; col < matrixSize; ++col)
            matSumResult[row][col] = matA[row][col] + matB[row][col];
    return NULL;
}

void *computeDiff(void *args)
{
    const int id = *(const int *)args;
    for (int row = id; row < matrixSize; row += NUM_THREADS)
        for (int col = 0; col < matrixSize; ++col)
            matDiffResult[row][col] = matA[row][col] - matB[row][col];
    return NULL;
}

/* Matrix product, not element-wise multiplication: C[i][j] = sum A[i][k]B[k][j]. */
void *computeProduct(void *args)
{
    const int id = *(const int *)args;
    for (int row = id; row < matrixSize; row += NUM_THREADS) {
        for (int col = 0; col < matrixSize; ++col) {
            int dotProduct = 0;
            for (int k = 0; k < matrixSize; ++k)
                dotProduct += matA[row][k] * matB[k][col];
            matProductResult[row][col] = dotProduct;
        }
    }
    return NULL;
}

/* One worker completes all three operations on its own non-overlapping rows. */
static void *computeAll(void *args)
{
    computeSum(args);
    computeDiff(args);
    computeProduct(args);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc > 2) {
        fprintf(stderr, "Usage: %s [matrix-size: 1..20]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 2) {
        char *end = NULL;
        errno = 0;
        long requested = strtol(argv[1], &end, 10);
        if (errno || end == argv[1] || *end != '\0' || requested < 1 || requested > MAX) {
            fprintf(stderr, "Matrix size must be an integer between 1 and %d\n", MAX);
            return EXIT_FAILURE;
        }
        matrixSize = (int)requested;
    }

    srand((unsigned)time(0));  /* Keep the starter template's random initialization. */
    fillMatrix(matA);
    fillMatrix(matB);

    printf("Matrix A:\n");
    printMatrix(matA);
    printf("Matrix B:\n");
    printMatrix(matB);

    pthread_t workers[NUM_THREADS];
    int ids[NUM_THREADS];
    int created = 0;
    for (int i = 0; i < NUM_THREADS; ++i) {
        ids[i] = i;
        int err = pthread_create(&workers[i], NULL, computeAll, &ids[i]);
        if (err != 0) {
            fprintf(stderr, "pthread_create failed for worker %d (error %d)\n", i, err);
            for (int j = 0; j < created; ++j) pthread_join(workers[j], NULL);
            return EXIT_FAILURE;
        }
        created++;
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
        int err = pthread_join(workers[i], NULL);
        if (err != 0) {
            fprintf(stderr, "pthread_join failed for worker %d (error %d)\n", i, err);
            return EXIT_FAILURE;
        }
    }

    printf("Results:\n");
    printf("Sum:\n");
    printMatrix(matSumResult);
    printf("Difference:\n");
    printMatrix(matDiffResult);
    printf("Product:\n");
    printMatrix(matProductResult);
    return EXIT_SUCCESS;
}
