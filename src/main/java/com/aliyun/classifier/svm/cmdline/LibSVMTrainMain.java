package com.aliyun.classifier.svm.cmdline;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.svm.LibSVMTrain;

public class LibSVMTrainMain extends Config {

    public static void main(String[] args) throws Exception {

        new LibSVMTrain().train();

        System.exit(0);

    }

}
