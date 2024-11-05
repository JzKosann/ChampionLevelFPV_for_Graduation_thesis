//
// Created by hg on 2024/9/13.
//

#include "rosprintf.h"
#include <ros/ros.h>

cRosInfo RosInfo;

void cRosInfo::printf(cRosInfo::eInfoLevel level, const char *__format, ...) {
    va_list args;
    va_start(args, __format);
    switch (level) {
        case eROS_Debug:
            ROS_DEBUG(__format, args);
            break;
        case eROS_Info:
            ROS_INFO(__format, args);
            break;
        case eROS_Warn:
            ROS_WARN(__format, args);
            break;
        case eROS_Error:
            ROS_ERROR(__format, args);
            break;
        case eROS_Fatal:
            ROS_FATAL(__format, args);
            break;
    }

    va_end(args);
}
