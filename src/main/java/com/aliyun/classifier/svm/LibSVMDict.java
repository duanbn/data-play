package com.aliyun.classifier.svm;

import java.io.File;
import java.io.FileWriter;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Corpus;
import com.aliyun.classifier.Feature;
import com.aliyun.classifier.fs.IGFeatureSelector;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;

public class LibSVMDict extends Config {

    public void run() throws Exception {
        Corpus corpus = new Corpus();

        // compute tf
        long start = System.currentTimeMillis();
        Multiset<String> tokenized = null;
        for (Map.Entry<String, Integer> categoryEntry : CATEGORY_NAME_CODE.entrySet()) {
            File[] categoryFiles = new File(CORPUS_DB, categoryEntry.getKey()).listFiles();
            corpus.categoryDocCount.put(categoryEntry.getKey(), categoryFiles.length);
            for (File corpusFile : categoryFiles) {

                tokenized = analysis(corpusFile);

                for (String token : tokenized.elementSet()) {
                    if (corpus.df.containsKey(token)) {
                        corpus.df.put(token, corpus.df.get(token) + 1);
                    } else {
                        corpus.df.put(token, 1);
                    }
                }

                corpus.featureDict.addAll(tokenized);

                corpus.categoryTokenized.put(categoryEntry.getKey(), tokenized.elementSet());

                corpus.N++;
            }
        }

        for (Multiset.Entry<String> entry : corpus.featureDict.entrySet()) {
            if (entry.getCount() > 2 && entry.getElement().length() > 1) {
                corpus.features.add(Feature.valueOf(entry.getElement()));
            }
        }

        System.out.println("collect feature done " + corpus.features.size() + " "
                + (System.currentTimeMillis() - start) + "ms");

        //            new CHIFeatureSelector(N, categoryTokenized, features).run();
        new IGFeatureSelector(corpus).run();

        Collections.sort(corpus.features, new Comparator<Feature>() {
            @Override
            public int compare(Feature o1, Feature o2) {
                if (o1.getQuality() == o2.getQuality()) {
                    return 0;
                }

                return o1.getQuality() > o2.getQuality() ? -1 : 1;
            }
        });

        List<String> lines = Lists.newArrayList(String.valueOf(corpus.N));
        long wordId = 0;
        StringBuilder line = null;
        for (Feature word : corpus.features) {
            wordId++;
            line = new StringBuilder();
            line.append(wordId).append(" ");
            line.append(word.getValue()).append(" ");
            line.append(corpus.df.get(word.getValue())).append(" ");
            line.append(word.getQuality());
            lines.add(line.toString());
        }

        try (FileWriter fw = new FileWriter(DICT)) {
            IOUtils.writeLines(lines, "\n", fw);
        }
    }

}
