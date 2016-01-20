package com.aliyun.classifier.svm.cmdline;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.svm.LibSVMOneTrain;

public class LibSVMOneTrainMain extends Config {

    public static void main(String[] args) throws Exception {

        new LibSVMOneTrain().run();

        System.exit(0);

    }

}
