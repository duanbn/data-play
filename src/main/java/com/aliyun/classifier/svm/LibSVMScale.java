package com.aliyun.classifier.svm;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;
import java.util.SortedSet;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Feature;
import com.google.common.collect.Lists;

public class LibSVMScale extends Config {

    public static final DecimalFormat df = new DecimalFormat("0.000000");

    public static void scale(SortedSet<Feature> doc, Range range) {
        double weight = 0, sum = 0;
        for (Feature feature : doc) {
            weight = feature.getWeight();

            sum = range.sum[feature.getId()][0];

            weight = weight / sum + feature.getChiScore();

            feature.setWeight(Double.parseDouble(df.format(weight)));
        }
    }

    public static Range materialize() {
        Range range = null;
        try (FileReader fr = new FileReader(RANGE)) {
            List<String> lines = IOUtils.readLines(fr);
            range = new Range(lines.size());
            String[] ss = null;
            for (String line : lines) {
                ss = line.split(" ");
                int featureId = Integer.parseInt(ss[0]);
                range.max[featureId][0] = Double.parseDouble(ss[1]);
                range.min[featureId][0] = Double.parseDouble(ss[2]);
                range.sum[featureId][0] = Double.parseDouble(ss[3]);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return range;
    }

    public static Range range(List<Feature> features, int maxId) {
        Range range = new Range(maxId);

        for (Feature feature : features) {
            range.max[feature.getId()][0] = Math.max(feature.getWeight(), range.max[feature.getId()][0]);
            range.min[feature.getId()][0] = Math.min(feature.getWeight(), range.min[feature.getId()][0]);
            range.sum[feature.getId()][0] += feature.getWeight();
        }

        List<String> lines = Lists.newArrayList();
        StringBuilder line = new StringBuilder();
        for (int i = 1; i < range.max.length; i++) {
            line.setLength(0);
            line.append(i).append(" ");
            line.append(range.min[i][0]).append(" ");
            line.append(range.max[i][0]).append(" ");
            line.append(range.sum[i][0]);
            lines.add(line.toString());
        }
        try (FileWriter fw = new FileWriter(RANGE)) {
            IOUtils.writeLines(lines, "\n", fw);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return range;
    }

    public static class Range {
        public final double[][] max;
        public final double[][] min;
        public final double[][] sum;

        public Range(int featureMaxId) {
            max = new double[featureMaxId + 1][1];
            min = new double[featureMaxId + 1][1];
            sum = new double[featureMaxId + 1][1];

            for (int i = 1; i < featureMaxId; i++) {
                max[i][0] = Double.MIN_VALUE;
                min[i][0] = Double.MAX_VALUE;
            }
        }
    }

}
