//
// Created by hg on 2024/9/13.
//

#ifndef SRC_ROSPRINTF_H
#define SRC_ROSPRINTF_H

#include <iostream>

class cRosInfo {
public:
    typedef enum {
        eROS_Debug,
        eROS_Info,
        eROS_Warn,
        eROS_Error,
        eROS_Fatal
    } eInfoLevel;
    eInfoLevel infoLevel;
    void printf(eInfoLevel level, const char *__restrict __format, ...);
};

extern cRosInfo RosInfo;
#endif //SRC_ROSPRINTF_H
