package com.aliyun.classifier.fs;

import java.util.Map;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CountDownLatch;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Corpus;
import com.aliyun.classifier.Feature;

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
                System.out.print(".");
            }
        }, 3000, 10 * 1000);

        long start = System.currentTimeMillis();
        for (Feature feature : corpus.features) {
            threadPool.submit(new ComputeTask(corpus, feature, HC, cdl));
        }

        cdl.await();
        timer.cancel();

        System.out.println("compute ig done " + (System.currentTimeMillis() - start) + "ms");

    }

    private static class ComputeTask implements Runnable {

        private Corpus         corpus;
        private Feature           feature;
        private double         HC;
        private CountDownLatch cdl;

        public ComputeTask(Corpus corpus, Feature feature, double HC, CountDownLatch cdl) {
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
                feature.setQuality(ig);
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
