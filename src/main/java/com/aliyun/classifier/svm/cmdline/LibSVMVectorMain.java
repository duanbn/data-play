package com.aliyun.classifier.svm.cmdline;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.CorpusVectorizer;

public class LibSVMVectorMain extends Config {

    public static void main(String[] args) throws Exception {

        new CorpusVectorizer().run();

        System.exit(0);

    }

}
