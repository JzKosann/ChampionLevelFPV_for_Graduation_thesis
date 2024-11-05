//
// Created by hg on 2024/9/24.
//

#ifndef SRC_CIMG_H
#define SRC_CIMG_H

#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ros/time.h>


class cImg {
private:
    cv::Mat img;
public:
    void imgGetTrain(const sensor_msgs::ImageConstPtr& msg);
};


#endif //SRC_CIMG_H
