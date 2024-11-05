//
// Created by hg on 2024/9/10.
//
/**
 * rosNode
 */

#include "rosInc.h"

using namespace std;

#define GPU_CUDA_INFO 0   //查看GPU信息
#define TRAIN_IMG_CAPTURE 1 //获取训练集代码


queue<sensor_msgs::ImageConstPtr> img_l_buf;
queue<sensor_msgs::ImageConstPtr> img_r_buf;
std::mutex m_buf;

/**
 * fpv位置获取，通过ros读取vins的topic来获取agent的世界坐标和四元数信息
 * @param msg
 * @type nav_msgs::Odometry
 */
static void fpvposeCallback(const nav_msgs::Odometry &msg) {
    fpvpose.loadpose(msg);
}

static void img_l_callback(const sensor_msgs::ImageConstPtr &msg) {
    m_buf.lock();
    img_l_buf.push(msg);
    m_buf.unlock();
}

static void img_r_callback(const sensor_msgs::ImageConstPtr &msg) {
    m_buf.lock();
    img_r_buf.push(msg);
    m_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}


//图像获取进程 同步左右两图像
void img_sync_process() {
    while (1) {
        cv::Mat image0, image1;
        std_msgs::Header header;
        double time = 0;
        m_buf.lock();
        if (!img_l_buf.empty() && !img_r_buf.empty()) {
            double time0 = img_l_buf.front()->header.stamp.toSec();
            double time1 = img_r_buf.front()->header.stamp.toSec();
            if (time0 < time1) {
                img_l_buf.pop();
                printf("throw img0\n");
            } else if (time0 > time1) {
                img_r_buf.pop();
                printf("throw img1\n");
            } else {
                time = img_l_buf.front()->header.stamp.toSec();
                header = img_l_buf.front()->header;
                image0 = getImageFromMsg(img_l_buf.front());
                img_l_buf.pop();
                image1 = getImageFromMsg(img_r_buf.front());
                img_r_buf.pop();
                //printf("find img0 and img1\n");
            }
        }
        m_buf.unlock();
        if (!image0.empty()) {
            cv::imshow("img0", image0);
            cv::imshow("img1", image1);
            cv::waitKey(0);
        }
            gateEstimator.inputImg(time, image0, image1);
    }
}

int main(int argc, char **argv) {   //argc=1 argv为文件地址
    //ROS
    setlocale(LC_ALL, "");

    ros::init(argc, argv, "fpv_node");
    ros::NodeHandle n("~");
//    printf("%d,%s\n", argc, *argv);
    //订阅vins-fusion-gpu的位恣信息
    ros::Subscriber fpvpose_sub = n.subscribe("/vins_fusion/odometry", 1000, fpvposeCallback);
    //TODO 订阅GateDetector推理出来的图像 一般都是在PY就同步好了?
    ros::Subscriber img_l_sub = n.subscribe("/GateDetector/detected_img_l", 100, img_l_callback);
    ros::Subscriber img_r_sub = n.subscribe("/GateDetector/detected_img_r", 100, img_r_callback);
//    ros::Subscriber fpvpose_sub = n.subscribe("/vins_fusion/odometry", 1000, fpvposeCallback);

#if GPU_CUDA_INFO  //检测CUDA信息
    cuda_test();
#endif
#if TRAIN_IMG_CAPTURE

#endif

//    cCnn gateDetector(1,
//                      2,
//                      {1});
//    gateDetector.run("/home/hg/Jinz/catkin_ws/src/fpv/src/gate_detector/LenaRGB.jpg");
    ROS_INFO("initialzed!");
    std::thread img_thread{img_sync_process};
    ros::spin();

    return 0;
}