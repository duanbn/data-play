package com.aliyun.classifier.svm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Word;
import com.google.common.collect.Lists;

public class LibSVMClassifier extends Config {

    private static final List<String>     categories = Lists.newArrayList();
    //    private static final Map<String, svm_model> models     = Maps.newTreeMap();
    private static svm_model              svmModel;

    private static final LibSVMClassifier instance   = new LibSVMClassifier();

    static {
        try (BufferedReader br = new BufferedReader(new FileReader(LABELINDEX))) {
            List<String> lines = IOUtils.readLines(br);
            String[] ss = null;
            for (String line : lines) {
                ss = line.split(" ");
                categories.add(ss[0]);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try {
            svmModel = svm.svm_load_model(new BufferedReader(new FileReader(getFile(CORPUS_NAME, "model"))));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        //        for (String category : categories) {
        //            try {
        //                File model = getFile(category, "model");
        //                models.put(category, svm.svm_load_model(new BufferedReader(new FileReader(model))));
        //                System.out.println("load " + model.getAbsolutePath());
        //            } catch (IOException e) {
        //                throw new RuntimeException(e);
        //            }
        //        }
    }

    private LibSVMClassifier() {
    }

    public static LibSVMClassifier getInstance() {
        return instance;
    }

    public String classify(String text) throws Exception {
        List<Word> words = LibSVMTextVectorizer.vectorization(text);
        svm_node[] x = new svm_node[words.size()];
        svm_node node = null;
        int index = 0;
        for (Word word : words) {
            node = new svm_node();
            node.index = (int) word.getId();
            node.value = word.getScore();
            x[index++] = node;
        }

        double v = svm.svm_predict(svmModel, x);

        //        List<String> result = Lists.newArrayList();
        //        for (Map.Entry<String, svm_model> entry : models.entrySet()) {
        //            double v = svm.svm_predict(entry.getValue(), x);
        //            if (v > 0) {
        //                result.add(entry.getKey());
        //            }
        //        }
        //        if (result.isEmpty()) {
        //            result.add("unknow");
        //        }

        return CATEGORY_CODE_NAME.get(new Double(v).intValue());
    }
}
