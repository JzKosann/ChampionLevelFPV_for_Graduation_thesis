//
// Created by hg on 2024/11/4.
//

#ifndef SRC_CGATE_ESTIMATOR_H
#define SRC_CGATE_ESTIMATOR_H

#include "rosInc.h"
#include <opencv2/core/types.hpp>
#include <eigen3/Eigen/Dense>
using namespace Eigen;
using namespace std;

class cgateEstimator {
private:
public:
    typedef int imgNum;
    typedef string imgname;
    map<imgNum, cv::Mat> GateDetected_Img;
    map<imgname, pair<vector<vector<Vector2d>>, vector<Vector2d>> > Detected_Corners;

    void inputImg(double time, const cv::Mat &img_l, const cv::Mat &img_r);

    pair<vector<vector<Vector2d>>, vector<Vector2d>> ImgProcess(cv::Mat img);//用pair来存储 <label对应的轮廓，四个corner质点>
};


#endif //SRC_CGATE_ESTIMATOR_H
