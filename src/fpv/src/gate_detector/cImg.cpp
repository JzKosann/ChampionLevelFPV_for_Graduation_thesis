//
// Created by hg on 2024/9/24.
//

#include "cImg.h"

cImg imgCamera;


long int count_ =0000;//不能命名成count
int n=0;

//void cImg::imgGetTrain(const sensor_msgs::ImageConstPtr& msg) {
//    cv_bridge::CvImagePtr cv_ptr;
//    try   //对错误异常进行捕获，检查数据的有效性
//    {
//        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
//        char base_name[256];
//        if(n%7==0)
//        {
//            sprintf(base_name,"./120H/%04ld.jpg",count_++);
//            //std::sprintf(base_name,"./image/%ld.jpg", msg->header.stamp.toNSec());//获取ROS时间戳
//            cv::imwrite(base_name, cv_ptr->image);
//            ROS_INFO_STREAM("Saved image to " << base_name);
//        }
//        n++;
//        std::cout<<"n= "<<n<<std::endl;
//    }
//    catch(cv_bridge::Exception& e)  //异常处理
//    {
//        ROS_ERROR("cv_bridge exception: %s", e.what());
//        return;
//    }
//       // 获取ROS时间戳并生成文件名
////      std::string filename =std::to_string(msg->header.stamp.toNSec()) + ".jpg";
////
////     // 保存图像到文件
////      cv::imwrite(filename, cv_ptr->image);
////     ROS_INFO_STREAM("Saved image to " << filename);
//}
