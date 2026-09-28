package com.rudra.visionv4;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class Yolo26Ncnn {

    public class Obj {
        public float x;
        public float y;
        public float w;
        public float h;
        public String label;
        public float prob;
    }

    public native boolean loadModel(AssetManager mgr, int modelid, int useGpu);
    public native Obj[] detect(Bitmap bitmap);

    static {
        System.loadLibrary("yolo26ncnn");
    }
}
