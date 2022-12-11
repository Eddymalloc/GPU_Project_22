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

__global__ void saxpy_monolithicc(int n, float a, float *x, float *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          y[i] = a * x[i] + y[i];
      }
}

float *vecAdd(float *A, float *B, int n)
{
    //defining the sizes of the memory objects to be copied to/from the device (in this case all three)
    int size = pow(2, 24);
    int blockSize = 32;
    int numBlocks = ceil(n/blockSize);

    // Allocate device memory for vectors A & B and the result vector C
    //1) declaring vectors
    float *d_A;
    float *d_B;
    float *d_C;

    //2) allocating memory for vectors
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Transfer the host input vectors A & B to device vectors d_A is used 
    // at the start, later d_C is used as an output as well
    //cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch the VecAdd CUDA Kernel with one thread for each element
    vecInitKernel <<<numBlocks, blockSize >>> (d_A, n, 1.0f);
    vecInitKernel <<<numBlocks, blockSize >>> (d_B, n, 2.0f);
    vecAddKernel <<<numBlocks, blockSize>>> (d_A, d_B, d_C, n);
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

  float maxError = 0.0f;
  for (int i = 0; i < size; i++)
    if (C[i] != 3.0f) {
        std::cout << "Max error: " << maxError << std::endl;
    }

    if (cudaerr != cudaSuccess)
        fprintf(stderr, "Failed to copy result vector from device to host (error code %s)!\n", cudaGetErrorString(cudaerr));

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

float *vecAdd2(float *A, float *B, int n)
{
    //defining the sizes of the memory objects to be copied to/from the device (in this case all three)
    int size = pow(2, 24);
    int blockSize = 32;
    int numBlocks = ceil(n/blockSize);

    // Allocate device memory for vectors A & B and the result vector C
    //1) declaring vectors
    float *d_A;
    float *d_B;
    float *d_C;

     // vector in host memory (declaration + allocation on host memory).
    float *C = (float*) malloc(size);

    //2) allocating memory for vectors
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Transfer the host input vectors A & B to device vectors d_A is used 
    // at the start, later d_C is used as an output as well
    //cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch the VecAdd CUDA Kernel with one thread for each element
    vecInitKernel <<<numBlocks, blockSize >>> (d_A, n, 1.0f);
    vecInitKernel <<<numBlocks, blockSize >>> (d_B, n, 2.0f);
    vecAddKernel <<<numBlocks, blockSize>>> (d_A, d_B, d_C, n);
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

  float maxError = 0.0f;
  for (int i = 0; i < size; i++)
    if (C[i] != 3.0f) {
        std::cout << "Max error: " << maxError << std::endl;
    }

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

    if (ac != 2) {
        std::cout << "non" << std::endl;
        return (84);
    }

    if (strcmp(av[1], "standard") == 0) {
    //for standard memory
		cudaMalloc(&x, N*sizeof(float));
        cudaMalloc(&x, N*sizeof(float));
    } else if(strcmp(av[1], "unified") == 0) {
    //only for unified memory
		cudaMallocManaged(&x, N*sizeof(float));
        cudaMallocManaged(&y, N*sizeof(float));
    } else {
        std::cout << "non" << std::endl;
        return 84;
    }

    c = vecAdd(x, y, N);

    std::cout << c << std::endl;
    
    return 0;
}