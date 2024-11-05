//
// Created by hg on 2024/11/4.
//

#include "cGate_Estimator.h"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include <opencv2/cudaarithm.hpp>
#include "rosInc.h"

using namespace cv;
cgateEstimator gateEstimator;
/**
 * @brief 输入一帧左右两相机的图像 分别进行处理取出图像坐标
 * @todo 滤波？
 * @param time
 * @param img_l
 * @param img_r
 */
void cgateEstimator::inputImg(double time, const cv::Mat &img_l, const cv::Mat &img_r) {
#if 1    //todo 暂时先采用轮廓取坐标平均 来进行处理
    pair<vector<vector<Vector2d >>, vector<Vector2d >> _img_l;
    pair<vector<vector<Vector2d >>, vector<Vector2d >> _img_r;
    _img_l = ImgProcess(img_l);
    _img_r = ImgProcess(img_r);
    Detected_Corners["left"] = _img_l;
    Detected_Corners["right"] = _img_r;

#endif
}

pair<vector<vector<Vector2d>>, vector<Vector2d>> cgateEstimator::ImgProcess(cv::Mat img) {
    pair<vector<vector<Vector2d >>, vector<Vector2d >> _img;
    cuda::GpuMat img_gpu, dst[4], dst_mat[4];
    img_gpu.upload(img);
    for (int i = 0; i < 3; i++)
        cv::cuda::threshold(img_gpu, dst[i], i, 255, cv::THRESH_BINARY);
    // 进行逐步相减
    for (int i = 0; i < 3; i++) {
        cv::cuda::subtract(dst[i], dst[i + 1], dst[i]);
    }
    cv::Mat dst_cpu[4];
    for (int i = 0; i <= 3; i++) {
        dst[i].download(dst_cpu[i]);
        for (int x = 0; x < img.rows; x++) {
            for (int y = 0; y < img.cols; y++) {
                Vector2d v(x, y);
                uchar pixelValue = dst_cpu[i].at<uchar>(y, x);
                if (pixelValue) {
                    _img.first[i].push_back(v); // 使用 pixelValue 作为索引
                    _img.second[i].x() += x;
                    _img.second[i].y() += y;
                }
            }
        }
        _img.second[i].x() /= _img.second[i].size();
        _img.second[i].y() /= _img.second[i].size();
    }


    return _img;
}


