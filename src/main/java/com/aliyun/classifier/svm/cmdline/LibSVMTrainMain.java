package com.aliyun.classifier.svm.cmdline;

import java.io.File;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.aliyun.classifier.CHIFeatureSelector;
import com.aliyun.classifier.Config;
import com.aliyun.classifier.Word;
import com.aliyun.classifier.svm.LibSVMDf;
import com.aliyun.classifier.svm.LibSVMTrain;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;

public class LibSVMTrainMain extends Config {

    public static void main(String[] args) throws Exception {

        if (args != null && args.length > 0 && Boolean.parseBoolean(args[0])) {
            int N = 0;
            Multimap<String, Set<String>> categoryTokenized = ArrayListMultimap.create();
            Multiset<String> featureDict = ConcurrentHashMultiset.create();

            // compute tf
            long start = System.currentTimeMillis();
            Multiset<String> tokenized = null;
            for (Map.Entry<String, Integer> categoryEntry : CATEGORY_NAME_CODE.entrySet()) {
                for (File corpusFile : new File(CORPUS_DB, categoryEntry.getKey()).listFiles()) {
                    tokenized = analysis(corpusFile);
                    featureDict.addAll(tokenized);
                    categoryTokenized.put(categoryEntry.getKey(), tokenized.elementSet());
                    N++;
                }
            }
            System.out.println("collect feature done " + featureDict.elementSet().size() + " "
                    + (System.currentTimeMillis() - start) + "ms");

            List<Word> featureWords = new CHIFeatureSelector().run(N, categoryTokenized, featureDict);

            new LibSVMDf().run(N, categoryTokenized, featureWords);
        }

        new LibSVMTrain().run();

        System.exit(0);

    }

}
