#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <vector>

#include <ncnn/platform.h>
#include <ncnn/benchmark.h>
#include <ncnn/cpu.h>

#include "yolo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static Yolo* g_yolo = nullptr;
static ncnn::Mutex lock;

static const int TARGET_SIZE = 640;
static const float MEAN_VALS[3] = {0.f, 0.f, 0.f};
static const float NORM_VALS[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

extern "C" {

//------------------------------------------------
// JNI LOAD
//------------------------------------------------
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "JNI Loaded");
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    ncnn::MutexLockGuard g(lock);
    delete g_yolo;
    g_yolo = nullptr;
}

//------------------------------------------------
// LOAD MODEL
//------------------------------------------------
JNIEXPORT jboolean JNICALL
Java_com_rudra_visionv4_Yolo26Ncnn_loadModel(
        JNIEnv* env,
        jobject thiz,
        jobject assetManager,
        jint modelid,
        jint useGpu)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    ncnn::MutexLockGuard g(lock);

    if (!g_yolo)
        g_yolo = new Yolo();

    int ret = g_yolo->load(
            mgr,
            "yolo26n",
            TARGET_SIZE,
            MEAN_VALS,
            NORM_VALS,
            useGpu == 1
    );

    __android_log_print(
            ANDROID_LOG_DEBUG,
            "Yolo26Ncnn",
            "Model load returned %d",
            ret
    );

    return ret == 0 ? JNI_TRUE : JNI_FALSE;
}

//------------------------------------------------
// DETECT
//------------------------------------------------
JNIEXPORT jobjectArray JNICALL
Java_com_rudra_visionv4_Yolo26Ncnn_detect(
        JNIEnv* env,
        jobject thiz,
        jobject bitmap)
{
    if (!g_yolo)
        return nullptr;

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    void* pixels = nullptr;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    // Bitmap -> OpenCV
    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);

    // Detection
    std::vector<Object> objects;
    g_yolo->detect(bgr, objects);

    // Draw boxes directly on bitmap
    g_yolo->draw(bgr, objects);

    // Copy back
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);

    AndroidBitmap_unlockPixels(env, bitmap);

    __android_log_print(
            ANDROID_LOG_DEBUG,
            "Yolo26Ncnn",
            "Detected %zu objects",
            objects.size()
    );

    // Java Obj class
    jclass cls = env->FindClass("com/rudra/visionv4/Yolo26Ncnn$Obj");

    jmethodID ctor = env->GetMethodID(
            cls,
            "<init>",
            "(Lcom/rudra/visionv4/Yolo26Ncnn;)V"
    );

    jfieldID xField = env->GetFieldID(cls, "x", "F");
    jfieldID yField = env->GetFieldID(cls, "y", "F");
    jfieldID wField = env->GetFieldID(cls, "w", "F");
    jfieldID hField = env->GetFieldID(cls, "h", "F");
    jfieldID labelField = env->GetFieldID(cls, "label", "Ljava/lang/String;");
    jfieldID probField = env->GetFieldID(cls, "prob", "F");

    jobjectArray array = env->NewObjectArray(objects.size(), cls, nullptr);

    for (size_t i = 0; i < objects.size(); i++)
    {
        jobject obj = env->NewObject(cls, ctor, thiz);

        env->SetFloatField(obj, xField, objects[i].rect.x);
        env->SetFloatField(obj, yField, objects[i].rect.y);
        env->SetFloatField(obj, wField, objects[i].rect.width);
        env->SetFloatField(obj, hField, objects[i].rect.height);

        env->SetObjectField(
                obj,
                labelField,
                env->NewStringUTF(class_names[objects[i].label])
        );

        env->SetFloatField(obj, probField, objects[i].prob);

        env->SetObjectArrayElement(array, i, obj);
        env->DeleteLocalRef(obj);
    }

    return array;
}

}