//
// Created by hg on 2024/9/9.
//

#ifndef SRC_FPVPOSE_H
#define SRC_FPVPOSE_H

#include <nav_msgs/Odometry.h>
#include <vector>
#include <eigen3/Eigen/Dense>
using namespace Eigen;

class cFpvpose {
private:
    //存储第一手数据
    struct {
        double x, y, z;
    } _stmpPointPos;
    struct {
        double x, y, z, w;
//        double[36] covariance;
    } _stmpQuaternion;

    double _odo_time;
public:
    void loadpose(const nav_msgs::Odometry &msg);//读取位置数据


};


#endif //SRC_FPVPOSE_H
