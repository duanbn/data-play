package com.aliyun.classifier.fs;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CountDownLatch;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Corpus;
import com.aliyun.classifier.Feature;
import com.google.common.collect.Lists;

public class IGFeatureSelector extends Config {

    private Corpus corpus;

    public IGFeatureSelector(Corpus corpus) {
        this.corpus = corpus;
    }

    public void run() throws Exception {
        double HC = 0;
        for (Map.Entry<String, Integer> entry : corpus.categoryDocCount.entrySet()) {
            HC += p(entry.getValue(), corpus.N) * log2(p(entry.getValue(), corpus.N));
        }

        final CountDownLatch cdl = new CountDownLatch(corpus.features.size());

        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                System.out.println(cdl.getCount() + "/" + corpus.features.size());
            }
        }, 3000, 10 * 1000);

        long start = System.currentTimeMillis();
        List<Feature> features = Lists.newCopyOnWriteArrayList();
        for (Feature feature : corpus.features) {
            threadPool.submit(new ComputeTask(features, corpus, feature, HC, cdl));
        }

        cdl.await();
        timer.cancel();

        corpus.features.clear();
        corpus.features.addAll(features);
        Collections.sort(corpus.features, new Comparator<Feature>() {
            @Override
            public int compare(Feature o1, Feature o2) {
                if (o1.getIgScore() == o2.getIgScore()) {
                    return 0;
                }
                return o1.getIgScore() > o2.getIgScore() ? -1 : 1;
            }
        });

        // do norm
        double max = corpus.features.get(0).getIgScore();
        double min = corpus.features.get(corpus.features.size() - 1).getIgScore();
        for (Feature feature : corpus.features) {
            if (max == min) {
                continue;
            }
            double normIg = (feature.getIgScore() - min) / (max - min);
            feature.setIgScore(normIg);
        }

        System.out.println("compute ig done " + (System.currentTimeMillis() - start) + "ms");
    }

    private static class ComputeTask implements Runnable {

        private List<Feature>  features;
        private Corpus         corpus;
        private Feature        feature;
        private double         HC;
        private CountDownLatch cdl;

        public ComputeTask(List<Feature> features, Corpus corpus, Feature feature, double HC, CountDownLatch cdl) {
            this.features = features;
            this.corpus = corpus;
            this.feature = feature;
            this.HC = HC;
            this.cdl = cdl;
        }

        @Override
        public void run() {
            try {
                double PT = 0, NPT = 0, TC = 0, NTC = 0;
                for (String category : corpus.categoryTokenized.keySet()) {
                    double A = 0, B = 0;
                    for (Set<String> doc : corpus.categoryTokenized.get(category)) {
                        if (doc.contains(feature.getValue())) {
                            A++;
                        } else {
                            B++;
                        }
                    }
                    TC += p(A, corpus.N) * log2(p(A, corpus.N));
                    NTC += p(B, corpus.N) * log2(p(B, corpus.N));
                }
                PT = p(corpus.df.get(feature.getValue()), corpus.N);
                NPT = p(corpus.N - corpus.df.get(feature.getValue()), corpus.N);

                double ig = -HC + (PT * TC) + (NPT * NTC);
                if (ig >= IG_THRESHOLD) {
                    feature.setIgScore(ig);
                    features.add(feature);
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                this.cdl.countDown();
            }
        }

    }

    private static double log2(double value) {
        if (value == 0) {
            return 0;
        }
        return Math.log(value) / Math.log(2.0);
    }

    private static double p(double value, double N) {
        return value / N;
    }

}
