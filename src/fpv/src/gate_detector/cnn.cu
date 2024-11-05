//
// Created by hg on 2024/9/11.
//
/**
 * 手写的cnn推理网络
 * 结合GPU
 */
#include "cnn.cuh"

//#include"cuda_runtime.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif
#define _3Dkernel 0

//#include <device_functions.h>
//声明一个类函数方便计算

/**
 * 构造函数-获取整个网络的大小
 * @param: layerNum 层数(中间层)
 * @param:
 * @brief:img size initialized 进行图片初始化 在这一步直接把CNN每一层的图片的大小计算好 减少后续并行计算的延迟
 */
cCnn::cCnn(int layerNum,    //层数
           int poolKerSize, //池化核
           std::vector<int> kerNum    //核数
) {
    _layerNum = layerNum;
    _poolKerSize = poolKerSize;
    imgStock[0] = cv::imread("/home/hg/Jinz/catkin_ws/src/fpv/src/gate_detector/LenaRGB.jpg");
    if (kerNum.size() != layerNum) {
        RosInfo.printf(cRosInfo::eROS_Error, "layerNum which u had inputted is not equal to kerNum array size!");
        printf("layerNum: %d\n kerNum: %zu\n", layerNum, kerNum.size());
        exit(0);
    }
    for (int i = 0; i < layerNum; i++)
        _convKerNum.push_back(kerNum[i]);
    printf("u had made a cnn-net made by JinZ\nnum of hidden layers: %d\n", layerNum);
    for (int i = 0; i < layerNum; i++) {
        printf("the %dth hidden layer has %d conv-kernels\n", i + 1, _convKerNum[i]);
        inImgSize[i].width = imgStock[i].cols;
        inImgSize[i].height = imgStock[i].rows;
        inImgSize[i].channel = imgStock[i].channels();
        convImgSize[i].width = imgStock[i].cols;
        convImgSize[i].height = imgStock[i].rows;
        convImgSize[i].channel = _convKerNum[i];
        poolImgSize[i].width = imgStock[i].cols / _poolKerSize;
        poolImgSize[i].height = imgStock[i].rows / _poolKerSize;
        poolImgSize[i].channel = convImgSize[i].channel;
    }
}

int cCnn::run(const std::string &filename) {
    cv::Mat inputImg = cv::imread(filename);
    getConvKernel();//读取卷积核参数
    //TODO:输入层
//    imgStock.insert(std::pair<layerSeq, cv::Mat>(0, inputImg));//输入图像
    //隐藏层
    hiddenLayer();
    cv::imshow("0", getHiddenLayerImg(0));
    cv::imshow("1", getHiddenLayerImg(1));
//    cv::imshow("2", getHiddenLayerImg(2));
    cv::waitKey(0);
    return 0;
};

int cCnn::hiddenLayer() {
    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < _layerNum; i++) {
        if (imgStock[i].empty()) {
            std::cout << "Could not open or find the image!" << std::endl;
            return -1;
        }

//        int channels = imgStock[i].channels();      //单张图像的通道
//        int conv_width = imgStock[i].cols;
//        int conv_height = imgStock[i].rows;
//        int pool_width = conv_width / _poolKerSize;
//        int pool_height = conv_height / _poolKerSize;
        // 分配 CUDA 内存
        unsigned char *img_input;//图像输入
        unsigned char *conv_output;//卷积层输出
        unsigned char *conv_cache;//申请全局缓存
        unsigned char *pool_output;//池化后输出
        size_t inputImgSize =
                imgStock[i].cols * imgStock[i].rows * imgStock[i].channels() * sizeof(unsigned char);   //卷积前的大小
        //卷积后的大小 长*宽*深度（卷积核数）
        size_t conv_outputImgSize = convImgSize[i].width * convImgSize[i].height *
                                    convImgSize[i].channel * sizeof(unsigned char);
        size_t outImgSize = poolImgSize[i].width * poolImgSize[i].height *
                            poolImgSize[i].channel * sizeof(unsigned char);
        //申请内存
        cudaMalloc((void **) &img_input, inputImgSize);
        cudaMalloc((void **) &conv_cache, inputImgSize);
        cudaMalloc((void **) &conv_output, conv_outputImgSize);
        cudaMalloc((void **) &pool_output, outImgSize);
        // 将图像数据从主机复制到设备
        cudaMemcpy(img_input, imgStock[i].data, inputImgSize, cudaMemcpyHostToDevice);
        // 申请卷积层和池化层对应的CUDA线程维度
        dim3 convblockDim(_convKerNum[i], convImgSize[i].channel);      //输出图像的通道数（卷积核数）*输入图像的通道数
        dim3 poolblockDim(_convKerNum[i], poolImgSize[i].channel);      //输出图像的通道数（卷积核数）*输入图像的通道数//TODO:有问题
        //线程块在线程网的索引 用识别图像长宽的二维线程网来处理，每个线程块处理一个像素的三个通道
        dim3 conv_gridDim(convImgSize[i].width, convImgSize[i].height);
        dim3 pool_gridDim(poolImgSize[i].width, poolImgSize[i].height);
        //获取卷积核 当前为第i层
        float *d_kernel;
        size_t kernelSize = convKernel[i].size() * sizeof(float);       //占用内存
        cudaMalloc((void **) &d_kernel, kernelSize);
        cudaMemcpy(d_kernel, convKernel[i].data(), kernelSize, cudaMemcpyHostToDevice);//读当前层的
        // 开始计时
        cudaEventRecord(start, 0);
        // 卷积层 + 激活层 conv->relu(max(value,0))
        globalConv<<<conv_gridDim, convblockDim>>>(img_input, conv_output, conv_cache,
                                                   inImgSize[i], convImgSize[i],
                                                   d_kernel);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in globalConv kernel: %s\n", cudaGetErrorString(err));
        }
        // 池化层 MaxPooling
//        globalMaxPool<<<pool_gridDim, poolblockDim>>>(conv_output, pool_output,
//                //池化输入图片的长、宽、/深度,深度为卷积层输出的（有多少个卷积核输出多少层）
//                                                      pool_width, pool_height, _convKerNum[i],
//                                                      _poolKerSize);
//        err = cudaGetLastError();
//        if (err != cudaSuccess) {
//            printf("CUDA error in globalMaxPool kernel: %s\n", cudaGetErrorString(err));
//        }
        // 停止计时
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);// 计算时间
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("the %dth hidden layer costs:%.2f ms\n", i + 1, milliseconds);
        cv::Mat output_img(convImgSize[i].height, convImgSize[i].width, CV_8UC1);//输出二维灰度图  //test
        cudaMemcpy(output_img.data, conv_output, conv_outputImgSize, cudaMemcpyDeviceToHost);
//        cv::Mat output_img(pool_height, pool_width, CV_8UC3);
//        cv::Mat output_img(pool_height, pool_width, CV_8UC1);//输出二维灰度图  //test
//        cudaMemcpy(output_img.data, pool_output, outImgSize, cudaMemcpyDeviceToHost);
        // 释放CUDA内存
        cudaFree(img_input);
        cudaFree(d_kernel);
        cudaFree(conv_output);
        cudaFree(pool_output);
        imgStock.insert(std::pair<layerSeq, cv::Mat>(i + 1, output_img));
    }
    return 0;
}

cv::Mat cCnn::getHiddenLayerImg(layerSeq seq) {
    return imgStock[seq];
}

int cCnn::getConvKernel() {
    std::ifstream file;
    file.open("/home/hg/Jinz/catkin_ws/src/fpv/src/gate_detector/modelparam.txt", std::ios::in);    //读取文件
    bool ret = file.good();
    if (ret)  RosInfo.printf(cRosInfo::eROS_Warn, "model is loading");
    else {
        RosInfo.printf(cRosInfo::eROS_Error, "cannot find out the model file, please check!");
        RosInfo.printf(cRosInfo::eROS_Error, "the program is stopped to prevent something boom");
        assert(file.is_open());
        return 0;
    }


    char c;
    int i = 0;
    while (!file.eof()) {
        bool is_firstLayer = true;//确认是否为第一层 因为第一层是
        if (is_firstLayer) {

        }
        file >> c;
        convKernel[0].push_back(std::atof(&c));
        i++;
    }
//    for (i = 0; i < 9; i++) {
//        printf("%.f\n", convKernel[0][0][i]);
//    }
    convKernel[0].pop_back();
    RosInfo.printf(cRosInfo::eROS_Info, "model be loaded!");
    return 1;
}
/**
 *
 * @param input     device_input 在设备端的图像的信息
 * @param output    device_output
 * @param width     图片的宽
 * @param height    图片的高
 * @param channels  色彩通道数
 * @param kernel    卷积核
 *
 * 三位数组转换为一维数组
 * img[a][b][c]= img[a * width * height + b * width + c];
 * blockDim 线程块各个维度的大小
 * blockIdx 当前线程块所处的线程格的坐标位置
 * threadIdx 线程所在的线程快的位置
 */
#define IMGWIDTH_MAX 640
#define IMGHEIGHT_MAX 480

//__global__ void
//globalConv(const unsigned char *input, unsigned char *output, unsigned char *cache,
//           int width, int height, int channels, int outChannel,
//           const float *kernel)
//{
//#if 0
//    int picSize = width * height * channels;    //单张图像像素的总数
//    //定位输出图像在线程处的坐标
//    int img_thX = blockIdx.x * blockDim.x + threadIdx.x;          //图像转为一维数组后在二维线程的坐标号
//    int img_thY = blockIdx.y * blockDim.y + threadIdx.y;
//    int pixelSeq = img_thY * blockDim.x * gridDim.x + img_thX;    //二维线程转为一维数组后的下标
//    int imgSeq = pixelSeq / picSize;        //当前是第imgSeq+1个图片
//    int imgOffset = imgSeq * picSize;
//    //由线程索引号映射原图数组坐标 当前所在三维坐标，通道数由参数传进来
//    int img_piC = (pixelSeq - imgOffset) % channels;                                    //当前所在原图象的通道
//    int img_piX = ((pixelSeq - imgOffset - img_piC) / channels) % width;              //当前所在原图像的x坐标
//    int img_piY = (((pixelSeq - imgOffset - img_piC) / channels) - img_piX) / width;  //当前所在原图像的y坐标
////    __syncthreads();
//    // 卷积核大小 (假设为 KERNEL_SIZE x KERNEL_SIZE x channels)
//    int kernelRadius = KERNEL_SIZE / 2;//
//    // 确保线程处理的坐标在图像范围内
//    if (img_piX >= width || img_piY >= height) {
//        printf("conv_(block(%d, %d),thread(%d, %d)) axis is out of range!\n",
//               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
//        return;
//    }
//    float pixelValue = 0.0f;    //像素值
//    // 遍历卷积核
//    for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
//        for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
//            int imgX = min(max((int) (img_piX + kx), 0), width - 1);  // 处理原图图像边界
//            int imgY = min(max((int) (img_piY + ky), 0), height - 1); // 处理原图图像边界
//            float kernelValue = kernel[((ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius)) * channels +
//                                       img_piC + imgSeq * KERNEL_SIZE * KERNEL_SIZE * channels];    //定位当前所在核的哪个参数
////            float kernelValue = kernel[(ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius)];//二维卷积
//            // 计算当前像素的卷积值 存入全局内存
//            cache[pixelSeq] += input[(imgY * width + imgX) * channels + img_piC] * kernelValue;
//        }
//    }
//    __syncthreads();//先处理同一卷积核 将三维图片合成一维图片
//    if (img_piC == 0)    //仅在深度为0的那个线程加和 避免数据冲突
//    {
//        for (int i = 0; i < channels; i++)
//        {
//            pixelValue += min(max((int) cache[imgOffset +  //图片的偏移量
//                                              (img_piY * width + img_piX) * channels + i], 0), 255 - 1);//单张图片内的坐标索引
//        }
//        output[(img_piY * width + img_piX) * outChannel +
//               imgSeq] = pixelValue;//输出到width*height*核数量（输出通道）（卷积层输入输出的图片尺寸一样）
//    }
////    output[(img_piY * width + img_piX) * outChannel +
////           imgSeq] = cache[pixelSeq];
//#else
//    /*
//     * 以output img的大小申请线程
//     * 线程索引就是输出图像的索引
//     * 要映射输出图像到输入图像
//     * 输出图像的通道数就是核数，第几通道就是第几核
//     * 要使用padding对边缘特征记录
//     * 输出图像不同通道的卷积对应的是不同的核
//     */
//    int picSize = width * height * channels;    //单张图像像素的总数
//    //定位输出图像在线程处的坐标
//    int img_thX = blockIdx.x * blockDim.x + threadIdx.x;          //图像转为一维数组后在二维线程的坐标号
//    int img_thY = blockIdx.y * blockDim.y + threadIdx.y;
//    int pixelSeq = img_thY * blockDim.x * gridDim.x + img_thX;    //二维线程转为一维数组后的下标
//    int imgSeq = pixelSeq / picSize;        //当前是第imgSeq+1个核
//    int imgOffset = imgSeq * picSize;       //计算第imgSeq+1个通道的索引偏移值
//    //由线程索引号映射原图数组坐标 当前所在三维坐标，通道数由参数传进来
//    int img_piC = (pixelSeq - imgOffset) % channels;                                    //当前所在原图象的通道
//    int img_piX = ((pixelSeq - imgOffset - img_piC) / channels) % width;              //当前所在原图像的x坐标
//    int img_piY = (((pixelSeq - imgOffset - img_piC) / channels) - img_piX) / width;  //当前所在原图像的y坐标
//    // 卷积核大小 (假设为 KERNEL_SIZE x KERNEL_SIZE x channels)
//    int kernelRadius = KERNEL_SIZE / 2;//
//    // 确保线程处理的坐标在图像范围内
//    if (img_piX >= width || img_piY >= height)
//    {
//        printf("conv_(block(%d, %d),thread(%d, %d)) axis is out of range!\n",
//               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
//        return;
//    }
//    float pixelValue = 0.0f;    //像素值
//    // 遍历卷积核
//    for (int kc = 0; kc <= channels; kc++)//这里是原图的通道
//    {
//        for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
//        {
//            for (int kx = -kernelRadius; kx <= kernelRadius; ++kx)
//            {
//                int imgX = min(max((int) (img_piX + kx), 0), width - 1);  // 处理原图图像边界
//                int imgY = min(max((int) (img_piY + ky), 0), height - 1); // 处理原图图像边界
//                float kernelValue = kernel[((ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius)) * channels +
//                                           kc + imgSeq * KERNEL_SIZE * KERNEL_SIZE * channels];    //定位当前所在核的哪个参数
////            float kernelValue = kernel[(ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius)];//二维卷积
//                // 计算当前像素的卷积值 存入全局内存
//                pixelValue += input[(imgY * width + imgX) * channels + kc] * kernelValue;   //这里直接遍历一整个核
//            }
//        }
//    }
//    output[(img_piY * width + img_piX) * outChannel + imgSeq] = pixelValue;
//
//#endif
//
//    __syncthreads();    //这里进行线程同步准备将全局内存的东西计算进去 放入width*height*kernelNum的三维数组
//
//}

__global__ void
globalConv(const unsigned char *input, unsigned char *output, unsigned char *cache,
           imgSize_t inImg, imgSize_t outImg,
           const float *kernel) {
#if 0
    int picSize = width * height * channels;    //单张图像像素的总数
//定位输出图像在线程处的坐标
int img_thX = blockIdx.x * blockDim.x + threadIdx.x;          //图像转为一维数组后在二维线程的坐标号
int img_thY = blockIdx.y * blockDim.y + threadIdx.y;
int pixelSeq = img_thY * blockDim.x * gridDim.x + img_thX;    //二维线程转为一维数组后的下标
int img_ker_seq = pixelSeq / picSize;        //当前是第imgSeq+1个图片
int imgOffset = img_ker_seq * picSize;
//由线程索引号映射原图数组坐标 当前所在三维坐标，通道数由参数传进来
int img_piC = (pixelSeq - imgOffset) % channels;                                    //当前所在原图象的通道
int img_piX = ((pixelSeq - imgOffset - img_piC) / channels) % width;              //当前所在原图像的x坐标
int img_piY = (((pixelSeq - imgOffset - img_piC) / channels) - img_piX) / width;  //当前所在原图像的y坐标
//    __syncthreads();
// 卷积核大小 (假设为 KERNEL_SIZE x KERNEL_SIZE x channels)
int kernelRadius = KERNEL_SIZE / 2;//
// 确保线程处理的坐标在图像范围内
if (img_piX >= width || img_piY >= height) {
    printf("conv_(block(%d, %d),thread(%d, %d)) axis is out of range!\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    return;
}
float pixelValue = 0.0f;    //像素值
// 遍历卷积核
for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
    for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
        int imgX = min(max((int) (img_piX + kx), 0), width - 1);  // 处理原图图像边界
        int imgY = min(max((int) (img_piY + ky), 0), height - 1); // 处理原图图像边界
        float kernelValue = kernel[((ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius)) * channels +
                                   img_piC + img_ker_seq * KERNEL_SIZE * KERNEL_SIZE * channels];    //定位当前所在核的哪个参数
//            float kernelValue = kernel[(ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius)];//二维卷积
        // 计算当前像素的卷积值 存入全局内存
        cache[pixelSeq] += input[(imgY * width + imgX) * channels + img_piC] * kernelValue;
    }
}
__syncthreads();//先处理同一卷积核 将三维图片合成一维图片
if (img_piC == 0)    //仅在深度为0的那个线程加和 避免数据冲突
{
    for (int i = 0; i < channels; i++)
    {
        pixelValue += min(max((int) cache[imgOffset +  //图片的偏移量
                                          (img_piY * width + img_piX) * channels + i], 0), 255 - 1);//单张图片内的坐标索引
    }
    output[(img_piY * width + img_piX) * outChannel +
           img_ker_seq] = pixelValue;//输出到width*height*核数量（输出通道）（卷积层输入输出的图片尺寸一样）
}
//    output[(img_piY * width + img_piX) * outChannel +
//           img_ker_seq] = cache[pixelSeq];
#else
    /*
     * 以output img的大小申请线程
     * 线程索引就是输出图像的索引
     * 要映射输出图像到输入图像
     * 输出图像的通道数就是核数，第几通道就是第几核
     * 要使用padding对边缘特征记录
     * 输出图像不同通道的卷积对应的是不同的核
     */
//    int picSize = inImg.width * inImg.height * inImg.channel;    //输入图像单张的大小
//    int picSize = inImg.width * inImg.height;    //输入图像单张的大小
    //定位输出图像在线程处的坐标
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int pixelSeq = blockId * (blockDim.x * blockDim.y) + threadId;    //二维线程转为一维数组后的下标
//    int img_ker_seq = pixelSeq / picSize;        //当前是第imgSeq+1个核
//    int imgOffset = img_ker_seq * picSize;       //计算第imgSeq+1个通道的索引偏移值   //output图像没有偏移的说法
    //由线程索引号映射原图数组坐标 当前所在三维坐标，通道数由参数传进来
    int img_piC = (pixelSeq) % outImg.channel;                                    //当前输出图像的通道 第img_piC个通道就是第img_piC+1个核
    int img_piX = ((pixelSeq  - img_piC) / outImg.channel) % outImg.width;              //当前输出图像的x坐标 对应了输入图像的x
    int img_piY = (((pixelSeq  - img_piC) / outImg.channel) - img_piX) / outImg.width;  //当前输出图像的y坐标 对应了输入图像的y
    // 卷积核大小 (假设为 KERNEL_SIZE x KERNEL_SIZE x channels)
    int kernelRadius = KERNEL_SIZE / 2;//
    // 确保线程处理的坐标在图像范围内
    if (img_piX >= inImg.width || img_piY >= inImg.height) {
        printf("conv_(block(%d, %d),thread(%d, %d)) axis is out of range!\n",
               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
        return;
    }
    float pixelValue = 0.0f;    //像素值
    // 遍历卷积核
    for (int kc = 0; kc <= inImg.channel; kc++)//这里是原图的通道
    {
        for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
            for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
                int imgX = min(max((int) (img_piX + kx), 0), inImg.width - 1);  // 处理原图图像边界
                int imgY = min(max((int) (img_piY + ky), 0), inImg.height - 1); // 处理原图图像边界
                float kernelValue = kernel[((ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius)) * inImg.channel +
                                           kc + img_piC * KERNEL_SIZE * KERNEL_SIZE * inImg.channel];    //定位当前所在核的哪个参数
//            float kernelValue = kernel[(ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius)];//二维卷积
                // 计算当前像素的卷积值 存入全局内存
                pixelValue += input[(imgY * inImg.width + imgX) * inImg.channel + kc] * kernelValue;   //这里直接遍历一整个核
            }
        }
    }
//    output[(img_piY * outImg.width + img_piX) * outImg.channel + img_ker_seq] = pixelValue;
    output[pixelSeq] = pixelValue;

#endif

    __syncthreads();    //这里进行线程同步准备将全局内存的东西计算进去 放入width*height*kernelNum的三维数组

}

/**
 * 最大池化 池化层
 * @param input             输入的是上一层处理得到的图像
 * @param output            输出图像
 * @param width             输出图像的宽度
 * @param height            输出图像的高度
 * @param channels          输入图像的色彩通道
 * @param maxPoolKerSize    默认池化核的步长和核大小相同
 */
__global__ void
globalMaxPool(const unsigned char *input, unsigned char *output, int width, int height, int channels,
              const int maxPoolKerSize) {    //TODO:这里可以在外面再封装一层函数把步长和申请内存池变量统一
//    printf("MaxPooling\n");
    //输出图像在进程里面二维的索引号
    int img_thX = blockIdx.x * blockDim.x + threadIdx.x;          //图像转为一维数组后在二维线程的坐标号
    int img_thY = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelSeq = img_thY * blockDim.x * gridDim.x + img_thX;    //二维线程转为一维数组后的下标
    //由线程索引号映射输出数组坐标或当前所在三维坐标，通道数由参数传进来
    int img_piC = (pixelSeq) % channels;                                    //当前所在输入图象的通道
    int img_piX = ((pixelSeq - img_piC) / channels) % width;              //当前所在输入图像的x坐标
    int img_piY = (((pixelSeq - img_piC) / channels) - img_piX) / width;  //当前所在输入图像的y坐标
    __syncthreads();
    // 确保线程处理的坐标在图像范围内
    if (img_piX >= width || img_piY >= height) {
        printf("pool_(block(%d, %d),thread(%d, %d)) axis is out of range!\n",
               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
        return;
    }
    float pixelValue = -FLT_MAX;    //像素值
    // 遍历卷积核 2*2
    for (int ky = 0; ky < maxPoolKerSize; ++ky) {
        for (int kx = 0; kx < maxPoolKerSize; ++kx) {
            int imgX = min(max(img_piX * maxPoolKerSize + kx, 0), width * maxPoolKerSize - 1);  // 处理边界
            int imgY = min(max(img_thY * maxPoolKerSize + ky, 0), height * maxPoolKerSize - 1); // 处理边界
            pixelValue = max(pixelValue, (float) input[(imgY * width + imgX) * channels + img_piC]);
        }
    }
//    printf("axis:%.d,%.d,%.d,axis value:%.f\n", out_x, out_y, c, pixelValue);
    __syncthreads();
    // 将结果存储到输出图像
    output[(img_piY * width + img_piX) * channels + img_piC] = min(max(int(pixelValue), 0), 255);
}
