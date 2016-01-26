package com.aliyun.classifier;

import java.io.File;
import java.io.FileWriter;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.fs.CHIFeatureSelector;
import com.aliyun.classifier.fs.IGFeatureSelector;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.common.collect.Sets;

public class CorpusDict extends Config {

    public void run() throws Exception {
        List<String> categoryLabelLines = Lists.newArrayList();
        for (Map.Entry<String, Integer> entry : CATEGORY_NAME_CODE.entrySet()) {
            categoryLabelLines.add(entry.getKey() + " " + entry.getValue());
        }
        try (FileWriter fw = new FileWriter(LABELINDEX)) {
            IOUtils.writeLines(categoryLabelLines, "\n", fw);
        }
        System.out.println("process category label index done");

        Corpus corpus = new Corpus();

        // compute tf
        long start = System.currentTimeMillis();
        Multiset<String> tokenized = null;
        for (Map.Entry<String, Integer> entry : CATEGORY_NAME_CODE.entrySet()) {
            String category = entry.getKey();

            File[] categoryFiles = new File(CORPUS_DB, category).listFiles();
            corpus.categoryDocCount.put(category, categoryFiles.length);

            for (File corpusFile : categoryFiles) {

                tokenized = analysis(corpusFile);

                for (String token : tokenized.elementSet()) {
                    if (corpus.df.containsKey(token)) {
                        corpus.df.put(token, corpus.df.get(token) + 1);
                    } else {
                        corpus.df.put(token, 1);
                    }
                }

                Multiset<String> features = corpus.categoryFeatures.get(category);
                if (features == null) {
                    features = ConcurrentHashMultiset.create();
                    corpus.categoryFeatures.put(category, features);
                }
                features.addAll(tokenized);

                corpus.categoryTokenized.put(category, tokenized.elementSet());

                corpus.N++;
            }
        }

        Set<Feature> featureSet = Sets.newLinkedHashSet();
        for (Multiset<String> features : corpus.categoryFeatures.values()) {
            for (Multiset.Entry<String> feature : features.entrySet()) {
                if (feature.getElement().length() > 1 && corpus.df.get(feature.getElement()) > 1) {
                    featureSet.add(Feature.valueOf(feature.getElement()));
                }
            }
        }
        corpus.features.addAll(featureSet);

        int featureId = 0;
        for (Feature feature : corpus.features) {
            featureId++;
            feature.setId(featureId);
            feature.setDf(corpus.df.get(feature.getValue()));
        }
        corpus.maxFeatureId = featureId;

        System.out.println("collect feature done " + corpus.features.size() + " "
                + (System.currentTimeMillis() - start) + "ms");

        corpus.tfPerCategory = new int[corpus.maxFeatureId + 1][CATEGORY_NAME_CODE.size() + 1];
        for (Feature feature : corpus.features) {
            for (Map.Entry<String, Multiset<String>> entry : corpus.categoryFeatures.entrySet()) {
                String category = entry.getKey();
                corpus.tfPerCategory[feature.getId()][CATEGORY_NAME_CODE.get(category)] = entry.getValue().count(
                        feature.getValue());
            }
        }

        new IGFeatureSelector(corpus).run();
        new CHIFeatureSelector(corpus).run();

        Iterator<Feature> featureIt = corpus.features.iterator();
        Feature item = null;
        while (featureIt.hasNext()) {
            item = featureIt.next();
            double score = item.getIgScore() * item.getChiScore();
            if (score < DR_THRESHOLD)
                featureIt.remove();
            else
                item.setScore(score);
        }
        Collections.sort(corpus.features, new Comparator<Feature>() {
            @Override
            public int compare(Feature o1, Feature o2) {
                if (o1.getScore() == o2.getScore()) {
                    return 0;
                }
                return o1.getScore() > o2.getScore() ? -1 : 1;
            }
        });

        List<String> featureLines = Lists.newArrayList();
        featureLines.add(String.valueOf(corpus.N) + "," + String.valueOf(corpus.maxFeatureId));
        List<String> tfLines = Lists.newArrayList();
        List<String> chiLines = Lists.newArrayList();
        StringBuilder line = new StringBuilder();
        for (Feature feature : corpus.features) {
            line.setLength(0);
            line.append(feature.getId()).append(" ");
            line.append(feature.getValue()).append(" ");
            line.append(feature.getDf()).append(" ");
            line.append(feature.getIgScore()).append(" ");
            line.append(feature.getChiScore()).append(" ");
            line.append(feature.getScore());
            featureLines.add(line.toString());

            line.setLength(0);
            line.append(feature.getId()).append(" ");
            int[] tfPerCategory = corpus.tfPerCategory[feature.getId()];
            for (int i = 1; i < tfPerCategory.length; i++) {
                line.append(tfPerCategory[i]).append(" ");
            }
            tfLines.add(line.toString());

            line.setLength(0);
            line.append(feature.getId()).append(" ");
            double[] chiPerCategory = corpus.chiPerCategory[feature.getId()];
            for (int i = 1; i < chiPerCategory.length; i++) {
                line.append(chiPerCategory[i]).append(" ");
            }
            chiLines.add(line.toString());
        }

        try (FileWriter fw = new FileWriter(DICT)) {
            IOUtils.writeLines(featureLines, "\n", fw);
        }
        try (FileWriter fw = new FileWriter(CATEGORYTF)) {
            IOUtils.writeLines(tfLines, "\n", fw);
        }
        try (FileWriter fw = new FileWriter(CHI)) {
            IOUtils.writeLines(chiLines, "\n", fw);
        }
    }
}
