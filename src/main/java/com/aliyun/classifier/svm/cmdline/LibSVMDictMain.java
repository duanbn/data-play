package com.aliyun.classifier.svm.cmdline;

import com.aliyun.classifier.CorpusDict;

public class LibSVMDictMain {

    public static void main(String[] args) throws Exception {

        new CorpusDict().run();

        System.exit(0);

    }

}
