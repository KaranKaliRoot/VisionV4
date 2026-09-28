// YOLO26 ncnn implementation
// out0: dims=2, w=8400, h=84  => [84 rows, 8400 cols]
// row 0..3 : cx, cy, w, h (decoded in 640x640 coords)
// row 4..83: 80 class probs (sigmoid already in graph)

#include "yolo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"
#include "layer.h"

#include <android/log.h>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <cmath>

const char* class_names[80] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};


#define TAG "YOLO26"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, (int)objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = (int)objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = objects[i].rect.area();

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float iou = union_area > 0.f ? (inter_area / union_area) : 0.f;

            if (iou > nms_threshold)
            {
                keep = 0;
                break;
            }
        }

        if (keep) picked.push_back(i);
    }
}

static void generate_proposals_yolo26(const ncnn::Mat& pred,
                                      float prob_threshold,
                                      std::vector<Object>& objects,
                                      float* out_global_max = nullptr)
{
    objects.clear();

    if (pred.dims != 2)
    {
        LOGD("generate_proposals: unexpected pred.dims=%d (expected 2)", pred.dims);
        if (out_global_max) *out_global_max = 0.f;
        return;
    }

    const int num_proposals = pred.w;        // 8400
    const int num_feat      = pred.h;        // 84
    const int num_class     = num_feat - 4;  // 80

    if (num_feat != 84)
    {
        LOGD("generate_proposals: unexpected pred.h=%d (expected 84=4+80)", num_feat);
        if (out_global_max) *out_global_max = 0.f;
        return;
    }

    const float* ptr_cx = pred.row(0);
    const float* ptr_cy = pred.row(1);
    const float* ptr_w  = pred.row(2);
    const float* ptr_h  = pred.row(3);

    float global_max = 0.f;

    for (int i = 0; i < num_proposals; i++)
    {
        int label = -1;
        float score = 0.f;

        for (int k = 0; k < num_class; k++)
        {
            const float* row_cls = pred.row(4 + k);
            float s = row_cls[i]; // already sigmoid
            if (s > score)
            {
                score = s;
                label = k;
            }
        }

        if (score > global_max) global_max = score;
        if (score < prob_threshold) continue;

        float cx = ptr_cx[i];
        float cy = ptr_cy[i];
        float bw = ptr_w[i];
        float bh = ptr_h[i];

        float x0 = cx - bw * 0.5f;
        float y0 = cy - bh * 0.5f;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width  = bw;
        obj.rect.height = bh;
        obj.label = label;
        obj.prob  = score;
        objects.push_back(obj);
    }

    __android_log_print(
    ANDROID_LOG_DEBUG,
    "YOLO26",
    "Global max confidence = %.6f",
    global_max
);

if (out_global_max) *out_global_max = global_max;
}

// helpers: detect cv::Mat channel format robustly
static int pick_pixel_type_for_ncnn(const cv::Mat& img)
{
    // Ultralytics expects RGB
    // OpenCV default is BGR for CV_8UC3
    // Camera/Bitmap pipelines sometimes yield RGBA CV_8UC4
    int type = img.type();
    if (type == CV_8UC3) return ncnn::Mat::PIXEL_BGR2RGB;
    if (type == CV_8UC4) return ncnn::Mat::PIXEL_RGBA2RGB;
    // if user already provides RGB
    if (type == CV_8UC1)
    {
        // not supported directly for YOLO; caller should convert to 3 channels
        return -1;
    }
    // fallback: assume BGR
    return ncnn::Mat::PIXEL_BGR2RGB;
}

Yolo::Yolo()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

Yolo::~Yolo()
{
    yolo.clear();
}
int Yolo::load(const char* modeltype,
               int _target_size,
               const float* _mean_vals,
               const float* _norm_vals,
               bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();
    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    char parampath[256];
    char modelpath[256];

    sprintf(parampath, "%s.ncnn.param", modeltype);
    sprintf(modelpath, "%s.ncnn.bin", modeltype);

    int p = yolo.load_param(parampath);
    int m = yolo.load_model(modelpath);

    LOGD("desktop load_param=%d load_model=%d", p, m);

    if (p != 0 || m != 0)
        return -1;

    target_size = _target_size;

    for (int i = 0; i < 3; i++)
    {
        mean_vals[i] = _mean_vals[i];
        norm_vals[i] = _norm_vals[i];
    }

    return 0;
}
int Yolo::load(AAssetManager* mgr,
               const char* modeltype,
               int _target_size,
               const float* _mean_vals,
               const float* _norm_vals,
               bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();
    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    char parampath[256];
    char modelpath[256];

    sprintf(parampath, "%s.ncnn.param", modeltype);
    sprintf(modelpath, "%s.ncnn.bin", modeltype);

    int p = yolo.load_param(mgr, parampath);
    int m = yolo.load_model(mgr, modelpath);

    LOGD("load_param=%d  load_model=%d", p, m);

    if (p != 0 || m != 0)
        return -1;

    target_size = _target_size;

    for (int i = 0; i < 3; i++)
    {
        mean_vals[i] = _mean_vals[i];
        norm_vals[i] = _norm_vals[i];
    }

    return 0;
}

int Yolo::detect(const cv::Mat& input, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    objects.clear();

    const int img_w = input.cols;
    const int img_h = input.rows;

    // Your model is fixed 640x640 (8400 points), so target_size MUST be 640
    const int dst_size = target_size; // set to 640 in Java/C++ init

    // letterbox scale to dst_size x dst_size
    float scale = std::min(dst_size / (float)img_w, dst_size / (float)img_h);
    int new_w = (int)std::round(img_w * scale);
    int new_h = (int)std::round(img_h * scale);

    int wpad = dst_size - new_w;
    int hpad = dst_size - new_h;
    int pad_left = wpad / 2;
    int pad_top  = hpad / 2;

    int pixel_type = pick_pixel_type_for_ncnn(input);
    if (pixel_type < 0)
    {
        LOGD("Unsupported input cv::Mat type=%d (expect CV_8UC3 or CV_8UC4)", input.type());
        return -1;
    }

    // Debug input
    LOGD("input: w=%d h=%d type=%d (CV_8UC3=%d CV_8UC4=%d)",
         img_w, img_h, input.type(), CV_8UC3, CV_8UC4);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            input.data,
            pixel_type,
            img_w, img_h,
            new_w, new_h
    );

    ncnn::Mat in_pad;
    ncnn::copy_make_border(
            in, in_pad,
            pad_top, hpad - pad_top,
            pad_left, wpad - pad_left,
            ncnn::BORDER_CONSTANT,
            114.f
    );

    // FORCE Ultralytics default: /255
    const float mean_vals_ultra[3] = {0.f, 0.f, 0.f};
    const float norm_vals_ultra[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(mean_vals_ultra, norm_vals_ultra);

    ncnn::Extractor ex = yolo.create_extractor();
    ex.set_light_mode(true);

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    LOGD("YOLO26 output: dims=%d, w=%d (proposals), h=%d (features), c=%d",
         out.dims, out.w, out.h, out.c);

    std::vector<Object> proposals;
    generate_proposals_yolo26(out, prob_threshold, proposals, nullptr);

    if (proposals.empty())
        return 0;

    // sort by score desc
    qsort_descent_inplace(proposals);

    // NMS (set nms_threshold<=0 to disable)
    std::vector<int> picked;
    if (nms_threshold > 0.f)
        nms_sorted_bboxes(proposals, picked, nms_threshold);
    else
    {
        picked.resize(proposals.size());
        for (int i = 0; i < (int)proposals.size(); i++) picked[i] = i;
    }

    LOGD("after NMS: %zu", picked.size());

    int count = (int)picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // Map from padded 640x640 coords back to original image coords
        float x0 = (objects[i].rect.x - (float)pad_left) / scale;
        float y0 = (objects[i].rect.y - (float)pad_top) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width  - (float)pad_left) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (float)pad_top) / scale;

        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width  = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort by area desc (optional)
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;

    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int Yolo::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const cv::Scalar colors[] = {
            cv::Scalar( 67,  54, 244), cv::Scalar( 30,  99, 233), cv::Scalar( 39, 176, 156),
            cv::Scalar( 58, 183, 103), cv::Scalar( 81, 181,  63), cv::Scalar(150, 243,  33),
            cv::Scalar(169, 244,   3), cv::Scalar(188, 212,   0), cv::Scalar(150, 136,   0),
            cv::Scalar(175,  80,  76), cv::Scalar(195,  74, 139), cv::Scalar(220,  57, 205),
            cv::Scalar(235,  59, 255), cv::Scalar(193,   7, 255), cv::Scalar(152,   0, 255),
            cv::Scalar( 87,  34, 255), cv::Scalar( 85,  72, 121), cv::Scalar(158, 158, 158),
            cv::Scalar(125, 139,  96)
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        const cv::Scalar& color = colors[i % 19];

        cv::rectangle(rgb, obj.rect, color, 2);

        char text[256];
        const char* label_name = (obj.label >= 0 && obj.label < 80) ? class_names[obj.label] : "unknown";
        sprintf(text, "%s %.1f%%", label_name, obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > rgb.cols) x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), color, -1);
        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }

    return 0;
}
