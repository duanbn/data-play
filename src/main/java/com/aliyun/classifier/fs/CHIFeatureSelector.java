package com.aliyun.classifier.fs;

import java.util.Arrays;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CountDownLatch;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Corpus;
import com.aliyun.classifier.Feature;
import com.aliyun.classifier.Math;

public class CHIFeatureSelector extends Config {

    private Corpus corpus;

    public CHIFeatureSelector(Corpus corpus) {
        this.corpus = corpus;
        corpus.chiPerCategory = new double[corpus.maxFeatureId + 1][CATEGORY_NAME_CODE.size() + 1];
    }

    public void run() throws Exception {
        final CountDownLatch cdl = new CountDownLatch(corpus.features.size());

        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                System.out.println(cdl.getCount() + "/" + corpus.features.size());
            }
        }, 3000, 10 * 1000);

        long start = System.currentTimeMillis();
        for (Feature feature : corpus.features) {
            threadPool.submit(new ComputeTask(corpus, CATEGORY_NAME_CODE.keySet(), feature, cdl));
        }
        cdl.await();
        timer.cancel();

        System.out.println("compute chi done " + (System.currentTimeMillis() - start) + "ms");
    }

    private static class ComputeTask implements Runnable {

        private Corpus         corpus;
        private Set<String>    categories;
        private Feature        feature;
        private CountDownLatch cdl;

        public ComputeTask(Corpus corpus, Set<String> categories, Feature feature, CountDownLatch cdl) {
            this.corpus = corpus;
            this.categories = categories;
            this.feature = feature;
            this.cdl = cdl;
        }

        public void run() {
            try {
                for (String category : categories) {
                    double chi = computeCategoryCHI(feature, category);
                    corpus.chiPerCategory[this.feature.getId()][CATEGORY_NAME_CODE.get(category)] = chi;
                }

                double[] chiScores = Arrays.copyOfRange(corpus.chiPerCategory[this.feature.getId()], 1,
                        corpus.chiPerCategory[this.feature.getId()].length);
                double chiScore = Math.sVariance(chiScores);
                this.feature.setChiScore(chiScore);
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                cdl.countDown();
            }
        }

        private double computeCategoryCHI(Feature feature, String category) {
            double A = 0, B = 0, C = 0, D = 0;
            for (String key : corpus.categoryTokenized.keySet()) {
                if (key.equals(category)) {
                    for (Set<String> docFeature : corpus.categoryTokenized.get(key)) {
                        if (docFeature.contains(feature.getValue())) {
                            A++;
                        } else {
                            C++;
                        }
                    }
                } else {
                    for (Set<String> docFeature : corpus.categoryTokenized.get(key)) {
                        if (docFeature.contains(feature.getValue())) {
                            B++;
                        } else {
                            D++;
                        }
                    }
                }
            }

            double numerator = corpus.N * java.lang.Math.pow(A * D - C * B, 2);
            double denominator = (A + B) * (A + C) * (B + D) * (C + D);

            if (numerator > 0.0 && denominator > 0.0) {
                return numerator / denominator;
            } else {
                return 0.0;
            }
        }
    }

}
