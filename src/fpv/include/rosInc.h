//
// Created by hg on 2024/9/10.
//

#ifndef SRC_ROSINC_H
#define SRC_ROSINC_H

/**
 * 功能头文件
 */
#include "ros/ros.h"
#include "../src/gate_detector/cImg.h"
#include "../src/gate_detector/cudatest.cuh"
#include "../src/gate_detector/cnn.cuh"
#include "../src/gate_detector/cGate_Estimator.h"
#include <opencv2/opencv.hpp>
#include <thread>


#include "rosprintf.h"
#include "../src/fpv_pose/fpvpose.h"
#include "../src/gate_detector/cGate_Estimator.h"

/**
 * 外部变量
 */
extern cFpvpose fpvpose;
extern cImg imgCamera;
extern cgateEstimator gateEstimator;
#endif //SRC_ROSINC_H
