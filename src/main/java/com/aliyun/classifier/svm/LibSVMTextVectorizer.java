package com.aliyun.classifier.svm;

import java.io.FileReader;
import java.io.StringReader;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Feature;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;
import com.google.common.collect.Sets;

public class LibSVMTextVectorizer extends Config {

    private static final Map<String, Feature> featureMap = Maps.newHashMap();

    private static int                        N;

    static {
        try (FileReader fr = new FileReader(DICT)) {
            String[] ss = null;
            List<String> lines = IOUtils.readLines(fr);
            N = Integer.parseInt(lines.get(0).split(",")[0]);
            Feature feature = null;
            for (String line : lines.subList(1, lines.size())) {
                ss = line.split(" +");
                int featureId = Integer.parseInt(ss[0]);
                String value = ss[1];
                int df = Integer.parseInt(ss[2]);
                double igScore = Double.parseDouble(ss[3]);
                double chiScore = Double.parseDouble(ss[4]);
                double score = Double.parseDouble(ss[5]);
                feature = Feature.valueOf(featureId, value, df, igScore, chiScore, score);
                featureMap.put(ss[1], feature);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static SortedSet<Feature> vectorization(String text) throws Exception {
        SortedSet<Feature> result = Sets.newTreeSet();
        Multiset<String> doc = analysis(new StringReader(text));

        double max = 0, min = 0;
        for (Multiset.Entry<String> word : doc.entrySet()) {
            if (!featureMap.containsKey(word.getElement())) {
                continue;
            }
            Feature feature = featureMap.get(word.getElement());
            double weight = weight(word.getCount(), feature.getDf(), N, feature.getScore());
            feature.setWeight(weight);
            max = java.lang.Math.max(weight, max);
            min = java.lang.Math.min(weight, min);
            result.add(feature);
        }
        for (Feature feature : result) {
            if (max == min) {
                continue;
            }
            double weight = (feature.getWeight() - min) / (max - min);
            feature.setWeight(weight);
        }
        return result;
    }
}
