#ifndef YOLO_H
#define YOLO_H

#include <opencv2/core/core.hpp>

#include "net.h"
#include "cpu.h"
#include "platform.h"

#include <vector>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class Yolo
{
public:
    Yolo();
    ~Yolo();

    int load(const char* modeltype,
             int target_size,
             const float* mean_vals,
             const float* norm_vals,
             bool use_gpu);

    int load(AAssetManager* mgr,
             const char* modeltype,
             int target_size,
             const float* mean_vals,
             const float* norm_vals,
             bool use_gpu);

    int detect(const cv::Mat& rgb,
               std::vector<Object>& objects,
               float prob_threshold = 0.25f,
               float nms_threshold = 0.45f);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

public:
    ncnn::Net yolo;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

extern const char* class_names[80];

#endif // YOLO_H