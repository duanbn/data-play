package com.aliyun.classifier.svm;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Feature;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public class LibSVMScale extends Config {

    public static final DecimalFormat df    = new DecimalFormat("0.000000");
    public static final double        lower = 0;
    public static final double        upper = 1;

    public static Range materialize(String category) {
        Range range = new Range();
        try (FileReader fr = new FileReader(getFile(category, "range"))) {
            List<String> lines = IOUtils.readLines(fr);
            String[] ss = null;
            for (String line : lines) {
                ss = line.split(" ");
                Feature word = Feature.valueOf(Long.parseLong(ss[0]));
                word.setMinScore(Double.parseDouble(ss[1]));
                word.setMaxScore(Double.parseDouble(ss[2]));
                range.put(word);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return range;
    }

    public static void serialize(Range range, String category) {
        Map<Long, Feature> data = range.getData();
        List<String> lines = Lists.newArrayList();
        for (Map.Entry<Long, Feature> entry : data.entrySet()) {
            StringBuilder line = new StringBuilder();
            line.append(entry.getValue().getId() + " " + entry.getValue().getMinScore() + " "
                    + entry.getValue().getMaxScore());
            lines.add(line.toString());
        }
        try (FileWriter fw = new FileWriter(getFile(category, "range"))) {
            IOUtils.writeLines(lines, "\n", fw);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void scale(List<Feature> doc, Range range) {
        double value = 0, max = 0, min = 0;
        Iterator<Feature> it = doc.iterator();
        Feature w = null;
        while (it.hasNext()) {
            w = it.next();
            value = w.getScore();
            Feature rangeWord = range.get(w.getId());
            if (rangeWord != null) {
                max = rangeWord.getMaxScore();
                min = rangeWord.getMinScore();

                if (max == min) {
                    continue;
                }

                if (value == min) {
                    value = lower;
                } else if (value == max) {
                    value = upper;
                } else {
                    value = lower + (upper - lower) * (value - min) / (max - min);
                }
            } else {
                value = 1;
            }

            if (value > 0) {
                w.setScore(Double.parseDouble(df.format(value)));
            } else {
                it.remove();
            }
        }
    }

    public static Range range(List<Feature> features, long maxWordId) {
        Range range = new Range();

        double[] max = new double[(int) maxWordId + 1];
        double[] min = new double[(int) maxWordId + 1];
        for (int i = 0; i <= maxWordId; i++) {
            max[i] = -Double.MAX_VALUE;
            min[i] = Double.MAX_VALUE;
        }
        for (Feature word : features) {
            max[(int) word.getId()] = Math.max(word.getScore(), max[(int) word.getId()]);
            min[(int) word.getId()] = Math.min(word.getScore(), min[(int) word.getId()]);
        }
        for (Feature word : features) {
            word.setMaxScore(Math.max(max[(int) word.getId()], 0));
            word.setMinScore(Math.min(min[(int) word.getId()], 0));
            range.put(word);
        }

        return range;
    }

    public static class Range {
        Map<Long, Feature> data = Maps.newTreeMap();

        public Map<Long, Feature> getData() {
            return this.data;
        }

        public void put(Feature word) {
            data.put(word.getId(), word);
        }

        public Feature get(long wordId) {
            return data.get(wordId);
        }

    }

}
