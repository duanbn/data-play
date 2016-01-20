package com.aliyun.classifier.fs;

import java.util.Map;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CountDownLatch;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Corpus;
import com.aliyun.classifier.Feature;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;

public class CHIFeatureSelector extends Config {

    private Corpus corpus;

    public CHIFeatureSelector(Corpus corpus) {
        this.corpus = corpus;
    }

    public void run() throws Exception {

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
            threadPool.submit(new ComputeTask(corpus.categoryTokenized, corpus.N, CATEGORY_NAME_CODE.keySet(), feature,
                    cdl));
        }
        cdl.await();
        timer.cancel();

        System.out.println("compute chi done " + (System.currentTimeMillis() - start) + "ms");
    }

    private static class ComputeTask implements Runnable {

        private Multimap<String, Set<String>> categoryTokenized;
        private int                           N;
        private Set<String>                   categories;
        private Feature                          feature;
        private CountDownLatch                cdl;

        public ComputeTask(Multimap<String, Set<String>> categoryTokenized, int N, Set<String> categories,
                           Feature feature, CountDownLatch cdl) {
            this.categoryTokenized = categoryTokenized;
            this.N = N;
            this.categories = categories;
            this.feature = feature;
            this.cdl = cdl;
        }

        public void run() {
            try {
                Map<String, Double> categoryCHI = Maps.newLinkedHashMap();
                for (String category : categories) {
                    double chi = computeCategoryCHI(feature, category);
                    categoryCHI.put(category, chi);
                }

                double featureCHI = normalization(categoryCHI);

                if (featureCHI > 0.0) {
                    this.feature.setQuality(featureCHI);
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                cdl.countDown();
            }
        }

        private double normalization(Map<String, Double> categoryCHI) {
            double sum = 0.0;
            for (Double value : categoryCHI.values()) {
                sum += value;
            }

            double max = 0.0, normValue = 0.0;
            for (Map.Entry<String, Double> entry : categoryCHI.entrySet()) {
                normValue = entry.getValue() / sum;
                max = max < normValue ? normValue : max;
                entry.setValue(normValue);
            }
            return max;
        }

        private double computeCategoryCHI(Feature feature, String category) {
            double A = 0, B = 0, C = 0, D = 0;
            for (String key : categoryTokenized.keySet()) {
                if (key.equals(category)) {
                    for (Set<String> doc : categoryTokenized.get(key)) {
                        if (doc.contains(feature.getValue())) {
                            A++;
                        } else {
                            C++;

                        }
                    }
                } else {
                    for (Set<String> docFeature : categoryTokenized.get(key)) {
                        if (docFeature.contains(feature.getValue())) {
                            B++;
                        } else {
                            D++;
                        }
                    }
                }
            }

            double numerator = N * Math.pow(A * D - C * B, 2);
            double denominator = (A + B) * (A + C) * (B + D) * (C + D);

            if (numerator > 0.0 && denominator > 0.0) {
                return numerator / denominator;
            } else {
                return 0.0;
            }
        }
    }

}
