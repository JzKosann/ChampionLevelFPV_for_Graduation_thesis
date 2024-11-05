//
// Created by hg on 2024/9/11.
//

#ifndef SRC_CNN_CUH
#define SRC_CNN_CUH

#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include "rosprintf.h"

//定义宏函数wbCheck，该函数用于检查Device内存是否分配成功，以避免写过多代码
#define cudaCheck(stmt)  do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf( "\n\nFailed to run stmt %d ", __LINE__);                       \
            printf( "Got CUDA error ...  %s \n\n", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                        \
        printf("successed\n");                        \
    } while(0)

#define KERNEL_SIZE 3

typedef int layerSeq;       //当前层索引
typedef int convKerSeq;       //核序号
typedef int kerSeq;       //核内索引
typedef int imgParam;

typedef struct {
    int width;
    int height;
    int channel;
} imgSize_t;

class cCnn {
public:

private:
    int _layerNum;  //层数
    std::vector<int> _convKerNum;    //卷积核数量
    int _poolKerSize;   //池化核大小
    cv::Mat _img;
    std::map<layerSeq, cv::Mat> imgStock;//图像储存池 格式（序号， 图像）
    std::map<layerSeq, std::vector<float>> convKernel;//卷积核
public:
    explicit cCnn(int layerNum, int poolKerSize, std::vector<int> kerNum);    //声明基本信息
    std::map<layerSeq, imgSize_t> inImgSize, convImgSize, poolImgSize;   //三种过程中需要用到的
    int run(cv::Mat &inputImg);  //执行函数 输入图像 结果？
    int run(const std::string &filename);  //执行函数 输入图像 结果？ TODO:暂时用读文件验证代码
    int getConvKernel();     //输出模型格式 假设为txt TODO:还有其他的模型格式
    int hiddenLayer();  //隐藏层

    cv::Mat getHiddenLayerImg(layerSeq seq);    //获取隐藏层中第seq层后输出的图像
};

int cnn_test();

//__global__ void
//globalConv(const unsigned char *input, unsigned char *output, unsigned char *cache,
//           int width, int height, int channels, int outChannel,
//           const float *kernel);
__global__ void
globalConv(const unsigned char *input, unsigned char *output, unsigned char *cache,
           imgSize_t inImg, imgSize_t outImg,
           const float *kernel);

__global__ void
globalMaxPool(const unsigned char *input, unsigned char *output, int width, int height, int channels,
              const int maxPoolKerSize);

#endif //SRC_CNN_CUH
