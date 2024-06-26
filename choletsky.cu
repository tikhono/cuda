#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <algorithm>
#include <vector>
#include <math.h>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
using std::generate;
using std::vector;

// CPU Functions on Integer Matrices
void runCholesky();
vector<int> generatePositiveDefiniteMat(int N);
vector<int> choleskyCPU(vector<int>& A, int N);
void compareCPUtoGPU(vector<int>& A, vector<int>& L, int N);
void printMatrix(vector<int>& L, int N);
void printTranspose(vector<int>& L, int N);

// GPU Functions on Integer Matrices
__global__ void choleskyColumn(int* A, int* L, int N, int colNum);
__global__ void choleskyDiag(int* A, int* L, int N, int colNum);
//__global__ void cholDecomp(int* A, int* L, int N);

// CPU Functions on Float Matrices
void runCholesky_fp();
vector<float> generatePositiveDefiniteMat_fp(int N);
vector<float> choleskyCPU_fp(vector<float>& A, int N);
void compareCPUtoGPU_fp(vector<float>& A, vector<float>& L, int N);
void printMatrix_fp(vector<float>& L, int N);
void printTranspose_fp(vector<float>& L, int N);

// GPU Functions on Float Matrices
__global__ void choleskyColumn_fp(float* A, float* L, int N, int colNum);
__global__ void choleskyDiag_fp(float* A, float* L, int N, int colNum);

int main() {

    // Cholesky with integer matrices
    runCholesky();

    // Cholesky with float matrices
  //  runCholesky_fp();

    return 0;
}

// Using Ints
void runCholesky() {

    cout << "Decomposing a matrix of ints..." << endl;
    cout << endl;

    // Number of rows/columns in the (square) input matrix - this parameter is used for a lot of other calculations
    int N = 3;

    // Size (in bytes) of matrix
    size_t bytes = N * N * sizeof(int);

    // Host matrices
    vector<int> h_A(N * N);
    vector<int> h_L(N * N, 1); // Fill h_L with 0s

    /* h_A Preset 1 (3x3): */
    /* h_A[0] = 4; h_A[1] = 12; h_A[2] = -16;
    h_A[3] = 12; h_A[4] = 37; h_A[5] = -43;
    h_A[6] = -16; h_A[7] = -43; h_A[8] = 98; */

    /* h_A Preset 2 (3x3): */
    h_A[0] = 25; h_A[1] = 15; h_A[2] = -5;
    h_A[3] = 15; h_A[4] = 18; h_A[5] = 0;
    h_A[6] = -5; h_A[7] = 0; h_A[8] = 11;

    /* h_A Preset 3 (4x4):  */
    /*h_A[0] = 16; h_A[1] = -12; h_A[2] = -12; h_A[3] = -16;
    h_A[4] = -12; h_A[5] = 25; h_A[6] = 1; h_A[7] = -4;
    h_A[8] = -12; h_A[9] = 1; h_A[10] = 17; h_A[11] = 14;
    h_A[12] = -16; h_A[13] = -4; h_A[14] = 14; h_A[15] = 57; */

    /* Or, Initialize matrix A as a random positive definite matrix */
    //h_A = generatePositiveDefiniteMat(N);

    cout << "Initial Matrix:" << endl;
    printMatrix(h_A, N);

    /* How the Cholesky decomposition is created:
    1. Get value in the first diagonal spot
    2. Decompose the rest of the values in that column beneath the diagonal
    3. Repeat steps 1 and 2 for the remainder of the columns
    */

    // Allocate device memory
    int* d_A, * d_L;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_L, bytes);

    // Copy data to the device
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per threadblock dimension
    int threads = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int blocks = 1;

    // Use dim3 structs for block and grid dimensions
    dim3 threadsPerBlock(threads, threads); // 32x32 threads per block - not all utilized though, this could be optimized
    int numBlocks = 1;

    // If N is large, can use this formula to get more blocks and blockDims:
            // int BLOCKS = N / THREADS
            // dim3 numBlocks(BLOCKS, BLOCKS);

    // Start timer for GPU code
    auto start_GPU = std::chrono::high_resolution_clock::now();

    /* Cholesky actually starts here: */
    // Call diag/column functions for the N columns
    for (int i = 0; i < N; i++) {
        // Launch diagonal kernel
        //choleskyDiag<<<1, 1>>>(d_A, d_L, N, i);
        choleskyDiag << <numBlocks, threadsPerBlock >> > (d_A, d_L, N, i);
        // Launch column kernel
        //choleskyColumn<<<1, 1>>>(d_A, d_L, N, i);
        choleskyColumn << <numBlocks, threadsPerBlock >> > (d_A, d_L, N, i);
    }
    cudaStreamSynchronize(0);
    // End timer
    auto stop_GPU = std::chrono::high_resolution_clock::now();

    // Copy data back to the host
    cudaMemcpy(h_L.data(), d_L, bytes, cudaMemcpyDeviceToHost);

    // Print resulting lower triangular matrix and its transpose
    cout << "Running Cholesky on the GPU:" << endl;
    cout << endl;
    cout << "Lower triangular decomposition:" << endl;
    printMatrix(h_L, N);
    /*cout << "L-transpose:" << endl;
    printTranspose(h_L, N);*/

    // Display execution time
    std::chrono::duration<double> time_GPU = stop_GPU - start_GPU;
    cout << "Execution time on GPU: " << fixed << time_GPU.count() << setprecision(6) << "s\n";
    cout << endl;

    cout << "Running Cholesky on the CPU:" << endl;
    cout << endl;
    // Check result by doing decomposition on the CPU and comparing results
    vector<int> L_CPU(N * N, 0);
    // Start timer for CPU code and run Cholesky
    auto start_CPU = std::chrono::high_resolution_clock::now();
    L_CPU = choleskyCPU(h_A, N);
    // End timer and display CPU execution time
    auto stop_CPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_CPU = stop_CPU - start_CPU;
    cout << "Lower triangular decomposition:" << endl;
    printMatrix(L_CPU, N);
    cout << "Execution time on CPU: " << fixed << time_CPU.count() << setprecision(6) << "s\n";

    // Check for errors in GPU code
    compareCPUtoGPU(h_L, L_CPU, N);

    // Free memory on device
    cudaFree(d_A);
    cudaFree(d_L);
}

// Calculate diagonal entries using Cholesky decomposition
__global__ void choleskyDiag(int* A, int* L, int N, int colNum) {

    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only dealing with the values on the diagonal
    if (row < N && col < N) {
        if (row == col && col == colNum) {
            int diagSum = 0;
            for (int k = 0; k < col; k++) {
                diagSum += (L[col * N + k]) * (L[col * N + k]);
            }
            L[row * N + col] = sqrt((float)(A[row * N + col] - diagSum));
        }
    }
}

// Calculate entries of a column using Cholesky decomposition
__global__ void choleskyColumn(int* A, int* L, int N, int colNum) {
   
    // Compute each thread's global row and column index
   // int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

   // L[col] = 2;
    //L[row * N + col] = 2;
   // L[0] = 23;
    // Only dealing with the column values below diagonal
    if (col == colNum && row > col && row < N) {
        int sum = 0;
        for (int k = 0; k < col; k++) {
            sum += (L[row * N + k] * L[col * N + k]);
        }
        L[row * N + col] = (A[row * N + col] - sum) / L[col * N + col];
    }
}

// Generates (most likely) a positive, definite matrix of size NxN to be decomposed
vector<int> generatePositiveDefiniteMat(int N) {

    // A = M * M_transpose -> generates A as a positive semidefinite matrix
    // Positive-valued diagonals -> positive definite matrix

    vector<int> h_M(N * N);
    vector<int> h_Mt(N * N);
    // Initialize matrix with random values
    for (int i = 0; i < N * N; i++) {
        h_M[i] = (rand() % 10) + 1;
    }
    // Transpose
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            h_Mt[i + j * N] = h_M[j + i * N];
        }
    }

    // Initialize matrix that will be returned
    vector<int> h_A(N * N, 0);
    // Multiply the two and store into new matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += h_M[i * N + k] * h_Mt[k * N + j];
            }
            h_A[i * N + j] = tmp;
        }
    }
    return h_A;
}

// Generate decomposition on the CPU for error-checking purposes
vector<int> choleskyCPU(vector<int>& A, int N) {

    vector<int> L_CPU(N * N, 0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            int sum = 0;
            // Diagonal
            if (j == i) {
                for (int k = 0; k < j; k++) {
                    sum += pow(L_CPU[j * N + k], 2);
                }
                L_CPU[j * N + j] = sqrt(A[j * N + j] - sum);
            }
            // Not Diagonal
            else {
                for (int k = 0; k < j; k++) {
                    sum += (L_CPU[i * N + k] * L_CPU[j * N + k]);
                }
                L_CPU[i * N + j] = (A[i * N + j] - sum) / L_CPU[j * N + j];
            }
        }
    }
    return L_CPU;
}

// Check GPU result against CPU - just comparing entries of the matrices
void compareCPUtoGPU(vector<int>& L_GPU, vector<int>& L_CPU, int N) {

    int errors = 0;
    int row, col;
    for (int i = 0; i < N * N; i++) {
        row = i * N;
        col = i;
        if (L_GPU[i] != L_CPU[i]) {
            errors += 1;
            printf("Row %d, Column %d:\n", row, col);
            printf("CPU Value: %d \t GPU Value: %d\n", L_CPU[i], L_GPU[i]);
        }
    }
    cout << "Total errors: " << errors << endl;
    cout << endl;
}

// Prints a matrix passed in as a 1-D vector
void printMatrix(vector<int>& L, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << L[i * N + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

// Prints the transpose of a matrix passed in as a 1-D vector
void printTranspose(vector<int>& L, int N) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            cout << L[j + i * N] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

// First draft of Cholesky GPU function - doesn't really work
/*
__global__ void cholDecomp(int* A, int* L, int N) {

    // Trying using shared memory - this failed big-time:

    //__shared__ int L[16];
    //__shared__ int A[16];

    // Copying data to shared memory
    //for (int i = 0; i < (N * N); i++) {
    //   A[i] = d_L[i];
    //   L[i] = d_L[i];
    //}
    //__syncthreads();



    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Checking boundaries
    if (row < N && col < N) {
        // Not on diagonal and in upper triangle - assign entries to be 0
        if (col > row) {
            L[row * N + col] = 0;
        }
        // On a diagonal
        else if (row == col) {
            //printf("On a diagonal.\n");
            //printf("Row=%d, Col=%d\n", row, col);
            int diagSum = 0;
            for (int k = 0; k < col; k++) {
                diagSum += (L[col * N + k]) * (L[col * N + k]);
            }
            //printf("Sum: %d\n", diagSum);
            L[row * N + col] = sqrt((float)(A[row * N + col] - diagSum));
        }
        // Not on a diagonal and in lower triangle
        else {
            int sum = 0;
            for (int k = 0; k < col; k++) {
                sum += (L[row * N + k] * L[col * N + k]);
            }
            L[row * N + col] = (A[row * N + col] - sum) / L[col * N + col];
        }
    }
}
*/


 /***************************************************************************************************Cholesky Decomposition Functions with Floats: ***************************************************************************************************************/

// Functions to run same code as above but with floats so I don't have to change everything up there
void runCholesky_fp() {

    cout << "Decomposing a matrix of floats..." << endl;
    cout << endl;

    int N = 6;
    size_t bytes = N * N * sizeof(int);

    vector<float> h_A(N * N);
    vector<float> h_L(N * N, 0.0f);

    /* h_A Preset 4 (4x4):  */
    /*h_A[0] = 18; h_A[1] = 22; h_A[2] = 54; h_A[3] = 42;
    h_A[4] = 22; h_A[5] = 70; h_A[6] = 86; h_A[7] = 62;
    h_A[8] = 54; h_A[9] = 86; h_A[10] = 174; h_A[11] = 134;
    h_A[12] = 42; h_A[13] = 62; h_A[14] = 134; h_A[15] = 106;*/

    // This would likely work better with floats:
    h_A = generatePositiveDefiniteMat_fp(N);

    cout << "Initial Matrix:" << endl;
    printMatrix_fp(h_A, N);

    float* d_A, * d_L;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_L, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 32;
    int blocks = 1;
    dim3 threadsPerBlock(threads, threads);
    int numBlocks = 1;

    // Start timer for GPU code
    auto start_GPU = std::chrono::high_resolution_clock::now();

    // Call diag/column functions for N columns
    for (int i = 0; i < N; i++) {
        // Launch diagonal kernel
        choleskyDiag_fp << <numBlocks, threadsPerBlock >> > (d_A, d_L, N, i);
        // Launch column kernel
        choleskyColumn_fp << <numBlocks, threadsPerBlock >> > (d_A, d_L, N, i);
    }

    auto stop_GPU = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_L.data(), d_L, bytes, cudaMemcpyDeviceToHost);

    cout << "Running Cholesky on the GPU:" << endl;
    cout << endl;
    cout << "Lower triangular decomposition:" << endl;
    printMatrix_fp(h_L, N);
    /*cout << "L-transpose:" << endl;
    printTranspose_fp(h_L, N);*/

    std::chrono::duration<double> time_GPU = stop_GPU - start_GPU;
    cout << "Execution time on GPU: " << fixed << time_GPU.count() << setprecision(6) << "s\n";
    cout << endl;

    cout << "Running Cholesky on the CPU:" << endl;
    cout << endl;
    vector<float> L_CPU(N * N, 0.0f);
    // Start timer for CPU code and run Cholesky
    auto start_CPU = std::chrono::high_resolution_clock::now();
    L_CPU = choleskyCPU_fp(h_A, N);
    // End timer and display CPU execution time
    auto stop_CPU = std::chrono::high_resolution_clock::now();
    cout << "Lower triangular decomposition:" << endl;
    printMatrix_fp(L_CPU, N);
    /*cout << "L-transpose:" << endl;
    printTranspose_fp(L_CPU, N);*/
    std::chrono::duration<double> time_CPU = stop_CPU - start_CPU;
    cout << "Execution time on CPU: " << fixed << time_CPU.count() << setprecision(6) << "s\n";

    // Free memory on device
    cudaFree(d_A);
    cudaFree(d_L);
}

__global__ void choleskyDiag_fp(float* A, float* L, int N, int colNum) {

    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only dealing with the values on the diagonal
    if (row < N && col < N) {
        if (row == col && col == colNum) {
            float diagSum = 0.0f;
            for (int k = 0; k < col; k++) {
                diagSum += (L[col * N + k]) * (L[col * N + k]);
            }
            L[row * N + col] = sqrt((float)(A[row * N + col] - diagSum));
        }
    }
}

__global__ void choleskyColumn_fp(float* A, float* L, int N, int colNum) {

    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only dealing with the column values below diagonal
    if (col == colNum && row > col && row < N) {
        float sum = 0.0f;
        for (int k = 0; k < col; k++) {
            sum += (L[row * N + k] * L[col * N + k]);
        }
        L[row * N + col] = (A[row * N + col] - sum) / L[col * N + col];
    }
}

vector<float> generatePositiveDefiniteMat_fp(int N) {
    vector<float> h_M(N * N);
    vector<float> h_Mt(N * N);
    // Initialize matrix with random values
    for (int i = 0; i < N * N; i++) {
        h_M[i] = (float)((rand() % 10) + 1);
    }
    // Transpose
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            h_Mt[i + j * N] = h_M[j + i * N];
        }
    }
    // Initialize matrix that will be returned
    vector<float> h_A(N * N, 0.0f);
    // Multiply the two and store into new matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += h_M[i * N + k] * h_Mt[k * N + j];
            }
            h_A[i * N + j] = tmp;
        }
    }
    return h_A;
}

vector<float> choleskyCPU_fp(vector<float>& A, int N) {

    vector<float> L_CPU(N * N, 0.0f);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            // Diagonal
            if (j == i) {
                for (int k = 0; k < j; k++) {
                    sum += pow(L_CPU[j * N + k], 2);
                }
                L_CPU[j * N + j] = sqrt(A[j * N + j] - sum);
            }
            // Not Diagonal
            else {
                for (int k = 0; k < j; k++) {
                    sum += (L_CPU[i * N + k] * L_CPU[j * N + k]);
                }
                L_CPU[i * N + j] = (A[i * N + j] - sum) / L_CPU[j * N + j];
            }
        }
    }
    return L_CPU;
}

void compareCPUtoGPU_fp(vector<float>& A, vector<float>& L, int N) {

    // FIXME: What constitues error in these fp numbers?

}

void printMatrix_fp(vector<float>& L, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << fixed << L[i * N + j] << setprecision(3) << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

void printTranspose_fp(vector<float>& L, int N) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            cout << L[j + i * N] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}
