//
// Created by hg on 2024/9/9.
//

#include "fpvpose.h"
#include <ros/ros.h>
#include <sstream>
#include "rosInc.h"

cFpvpose fpvpose;

void cFpvpose::loadpose(const nav_msgs::Odometry &msg) {
    _odo_time = msg.header.stamp.toSec();

    _stmpPointPos.x = msg.pose.pose.position.x;
    _stmpPointPos.y = msg.pose.pose.position.y;
    _stmpPointPos.z = msg.pose.pose.position.z;

    _stmpQuaternion.x = msg.pose.pose.orientation.x;
    _stmpQuaternion.y = msg.pose.pose.orientation.y;
    _stmpQuaternion.z = msg.pose.pose.orientation.z;
    _stmpQuaternion.w = msg.pose.pose.orientation.w;

    printf("time: %f, t: %f %f %f q: %f %f %f %f \n", _odo_time, _stmpPointPos.x,_stmpPointPos.y, _stmpPointPos.z,
           _stmpQuaternion.w, _stmpQuaternion.x, _stmpQuaternion.y, _stmpQuaternion.z);
}

