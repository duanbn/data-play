package com.aliyun.classifier.svm.cmdline;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.svm.LibSVMTrain;

public class LibSVMTrainMain extends Config {

    public static void main(String[] args) throws Exception {

        new LibSVMTrain().run();

        System.exit(0);

    }

}
