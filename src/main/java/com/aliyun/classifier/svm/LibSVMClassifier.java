package com.aliyun.classifier.svm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.SortedSet;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Feature;
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

    //    public String[] classifyFull(String domain, String text) throws Exception {
    //        List<String> result = Lists.newArrayList();
    //        for (Map.Entry<String, svm_model> entry : models.entrySet()) {
    //            double v = svm.svm_predict(entry.getValue(), getX(domain, text));
    //            if (v > 0) {
    //                result.add(entry.getKey());
    //            }
    //        }
    //        if (result.isEmpty()) {
    //            result.add("unknow");
    //        }
    //
    //        return result.toArray(new String[result.size()]);
    //    }

    public String classify(String domain, String text) throws Exception {

        double v = svm.svm_predict(svmModel, getX(domain, text));

        return CATEGORY_CODE_NAME.get(new Double(v).intValue());
    }

    private svm_node[] getX(String domain, String text) throws Exception {
        SortedSet<Feature> words = LibSVMTextVectorizer.vectorization(text);
        System.out.println(domain + " " + words);
        svm_node[] x = new svm_node[words.size()];
        svm_node node = null;
        int index = 0;
        for (Feature word : words) {
            node = new svm_node();
            node.index = (int) word.getId();
            node.value = word.getWeight();
            x[index++] = node;
        }
        return x;
    }
}
