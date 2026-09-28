cmake_minimum_required(VERSION 3.22.1)
project(yolo26ncnn)

# ---------- NCNN ----------
add_library(ncnn SHARED IMPORTED)

set_target_properties(ncnn PROPERTIES
    IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/libncnn.so
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/include/ncnn
)

# ---------- OpenCV ----------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv-mobile-4.10.0-android/sdk/native/jni/include
)

set(OpenCV_LIB_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv-mobile-4.10.0-android/sdk/native/staticlibs/${ANDROID_ABI}
)

# ---------- Your library ----------
add_library(yolo26ncnn SHARED
    yolo.cpp
    yolo26ncnn.cpp
)

target_link_libraries(yolo26ncnn
    ncnn

    ${OpenCV_LIB_DIR}/libopencv_core.a
    ${OpenCV_LIB_DIR}/libopencv_imgproc.a

    android
    log
    jnigraphics
    camera2ndk
    mediandk
)