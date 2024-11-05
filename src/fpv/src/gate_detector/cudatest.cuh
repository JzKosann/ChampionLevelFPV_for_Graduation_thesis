//
// Created by hg on 2024/9/11.
//

#ifndef SRC_CUDATEST_CUH
#define SRC_CUDATEST_CUH
//这段代码在.cuh头文件中
#include "cuda_runtime.h"

extern "C"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
int cuda_test();

#endif //SRC_CUDATEST_CUH
