//
// Created by hg on 2024/9/11.
//
//#include "cuda_runtime.h"
#include <stdio.h>


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
extern "C"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int getThreadNum()
{
    cudaDeviceProp prop{};
    int count;

    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("gpu num %d\n", count);
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: %d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}

int cuda_test(){
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//           c[0], c[1], c[2], c[3], c[4]);
    printf("cuda template is available！\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    getThreadNum();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int dev;
    for (dev = 0; dev < deviceCount; dev++)
    {
        int driver_version(0), runtime_version(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev == 0)
            if (deviceProp.minor = 9999 && deviceProp.major == 9999)
                printf("\n");

        printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driver_version);
        printf("CUDA Driver Version:                            %d.%d\n",
               driver_version / 1000, (driver_version % 1000) / 10);
        cudaRuntimeGetVersion(&runtime_version);
        printf("CUDA Runtime Version:                           %d.%d\n",
               runtime_version / 1000, (runtime_version % 1000) / 10);
        printf("Device Prop:                                    %d.%d\n",
               deviceProp.major, deviceProp.minor);
        printf("Total amount of Global Memory:                  %zu bytes\n",
               deviceProp.totalGlobalMem);
        printf("Number of SMs:                                  %d\n",
               deviceProp.multiProcessorCount);
        printf("Total amount of Constant Memory:                %zu bytes\n",
               deviceProp.totalConstMem);
        printf("Total amount of Shared Memory per block:        %zu bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("Total number of registers available per block:  %d\n",
               deviceProp.regsPerBlock);
        printf("Warp size:                                      %d\n",
               deviceProp.warpSize);
        printf("Maximum number of threads per SM:               %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block:            %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("Maximum size of each dimension of a block:      %d x %d x %d\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid:       %d x %d x %d\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum memory pitch:                           %zu bytes\n",
               deviceProp.memPitch);
        printf("Texture alignmemt:                              %zu bytes\n",
               deviceProp.texturePitchAlignment);
        printf("Clock rate:                                     %.2f GHz\n",
               (float)deviceProp.clockRate * 1e-6f);
        printf("Memory Clock rate:                              %.0f MHz\n",
               (float)deviceProp.memoryClockRate * 1e-3f);
        printf("Memory Bus Width:                               %d-bit\n",
               deviceProp.memoryBusWidth);
    }
    return 0;
}