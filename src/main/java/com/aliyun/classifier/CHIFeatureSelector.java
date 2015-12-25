package com.aliyun.classifier;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CountDownLatch;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;

public class CHIFeatureSelector extends Config {

    public List<Word> run(int N, Multimap<String, Set<String>> categoryTokenized, Multiset<String> featureDict)
            throws Exception {

        List<Word> result = Lists.newCopyOnWriteArrayList();

        for (Multiset.Entry<String> entry : featureDict.entrySet()) {
            if (entry.getCount() > 2 && entry.getElement().length() > 1) {
                result.add(Word.valueOf(entry.getElement()));
            }
        }

        final CountDownLatch cdl = new CountDownLatch(result.size());
        System.out.println("document tokenized done. feature size " + cdl.getCount());

        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                System.out.println("remain " + cdl.getCount());
            }
        }, 3000, 10 * 1000);

        long start = System.currentTimeMillis();

        for (Word feature : result) {
            threadPool.submit(new ComputeTask(categoryTokenized, N, CATEGORY_NAME_CODE.keySet(), feature, cdl));
        }
        cdl.await();
        System.out.println("done " + (System.currentTimeMillis() - start) + "ms");

        return result;
    }

    private static class ComputeTask implements Runnable {

        private Multimap<String, Set<String>> categoryTokenized;
        private int                           N;
        private Set<String>                   categories;
        private Word                          feature;
        private CountDownLatch                cdl;

        public ComputeTask(Multimap<String, Set<String>> categoryTokenized, int N, Set<String> categories,
                           Word feature, CountDownLatch cdl) {
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

        private double computeCategoryCHI(Word feature, String category) {
            double A = 0, B = 0, C = 0, D = 0;
            for (String key : categoryTokenized.keySet()) {
                if (key.equals(category)) {
                    for (Set<String> docFeature : categoryTokenized.get(key)) {
                        if (docFeature.contains(feature.getValue())) {
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
