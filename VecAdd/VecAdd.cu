#include <iostream>
#include <cuda.h>
#include <math.h>

__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
        C[i] = A[i] + B[i];
}

__global__ void vecInitKernel(float *A, int n, float value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
        A[i] = value;
}

__global__ void vecAddMonolithic(float a, float *x, float *y, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
      {
        y[i] = a * x[i] + y[i];
      }
}

float *vecAddStandard(float *d_A, float *d_B, int n)
{
    //defining the sizes of the memory objects to be copied to/from the device (in this case all three)
    int size = pow(2, 24);
    int blockSize = 32;
    int numBlocks = ceil(n/blockSize);

    // Allocate device memory for vectors A & B and the result vector C
    //1) declaring vectors
    float *d_C;

    //2) allocating memory for vectors (unified)
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc((void**)&d_C, size);

    //initializing the vectors with initKernel and with correct values
    vecInitKernel <<<numBlocks, blockSize >>> (d_A, n, 1.0f);
    vecInitKernel <<<numBlocks, blockSize >>> (d_B, n, 2.0f);
    vecInitKernel <<<numBlocks, blockSize >>> (d_C, n, 0.0f);

    // Launch the VecAdd CUDA Kernel with one thread for each element
    vecAddKernel <<<numBlocks, blockSize>>> (d_A, d_B, d_C, n);
    vecAddMonolithic <<<numBlocks, blockSize >>> (1.0f, d_B, d_C, n);

    // Check for errors launching the kernel
    cudaError_t cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess)
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(cudaerr));

    // Check for errors relecting kernel execution
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        fprintf(stderr, "Failed to semkinud kernel (error code %s)!\n", cudaGetErrorString(cudaerr));

    // Copy the device result vector in device memory to the host result
    // vector in host memory.
    float *C = (float*) malloc(size);
    cudaerr = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    //error displaying for vector addition
//   float maxError = 0.0f;
//   for (int i = 0; i < size; i++)
//     if (C[i] != 3.0f) {
//         std::cout << "Max error: " << maxError << std::endl;
//     }

    if (cudaerr != cudaSuccess)
        fprintf(stderr, "Failed to copy result vector from device to host (error code %s)!\n", cudaGetErrorString(cudaerr));

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

float *vecAddUnified(float *d_A, float *d_B, int n)
{
    //defining the sizes of the memory objects to be copied to/from the device (in this case all three)
    int size = pow(2, 24);
    int blockSize = 32;
    int numBlocks = ceil(n/blockSize);

    // Allocate device memory for result vector C
    //1) declaring vectors
    float *d_C;

     // vector in host memory (declaration + allocation on host memory).
    float *C = (float*) malloc(size);

    //2) allocating memory for vectors (unified)
    cudaMallocManaged(&d_A, size);
    cudaMallocManaged(&d_B, size);
    cudaMallocManaged((void**)&d_C, size);

    //initializing the vectors with initKernel
    vecInitKernel <<<numBlocks, blockSize >>> (d_A, n, 1.0f);
    vecInitKernel <<<numBlocks, blockSize >>> (d_B, n, 2.0f);
    vecInitKernel <<<numBlocks, blockSize >>> (d_C, n, 0.0f);

    // Launch the VecAdd CUDA Kernel with one thread for each element
    vecAddKernel <<<numBlocks, blockSize >>> (d_A, d_B, d_C, n);
    vecAddMonolithic <<<numBlocks, blockSize >>> (1.0f, d_B, d_C, n);

    // Check for errors launching the kernel
    cudaError_t cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess)
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(cudaerr));

    // Check for errors relecting kernel execution
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        fprintf(stderr, "Failed to semkinud kernel (error code %s)!\n", cudaGetErrorString(cudaerr));

    // Copy the device result vector in device memory to the host result
    cudaerr = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    //error display for vector addition
//   float maxError = 0.0f;
//   for (int i = 0; i < size; i++)
//     if (C[i] != 3.0f) {
//         std::cout << "Max error: " << maxError << std::endl;
//     }

    if (cudaerr != cudaSuccess)
        fprintf(stderr, "Failed to copy result vector from device to host (error code %s)!\n", cudaGetErrorString(cudaerr));

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

int main(int ac, char **av)
{
    int N = 1<<20;
    float *x, *y, *c;

    if (strcmp(av[1], "standard") == 0) {
    //for standard memory
		cudaMalloc(&x, N*sizeof(float));
        cudaMalloc(&y, N*sizeof(float));
        cudaMalloc(&c, N*sizeof(float));
        c = vecAddStandard(x, y, N);
    } else if(strcmp(av[1], "unified") == 0) {
    //for unified memory<
		cudaMallocManaged(&x, N*sizeof(float));
        cudaMallocManaged(&y, N*sizeof(float));
        cudaMallocManaged(&c, N*sizeof(float));
        c = vecAddUnified(x, y, N);
    } else {
        std::cout << "wrong arguments" << std::endl;
        return 84;
    }
    
    return 0;
}