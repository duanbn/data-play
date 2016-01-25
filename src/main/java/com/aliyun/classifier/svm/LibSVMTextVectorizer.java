package com.aliyun.classifier.svm;

import java.io.FileReader;
import java.io.StringReader;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Feature;
import com.aliyun.classifier.svm.LibSVMScale.Range;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;
import com.google.common.collect.Sets;

public class LibSVMTextVectorizer extends Config {

    private static final Range                range;
    private static final Map<String, Feature> featureMap = Maps.newHashMap();

    private static int                        N;

    static {
        range = LibSVMScale.materialize();

        try (FileReader fr = new FileReader(DICT)) {
            String[] ss = null;
            List<String> lines = IOUtils.readLines(fr);
            N = Integer.parseInt(lines.get(0));
            Feature feature = null;
            for (String line : lines.subList(1, lines.size())) {
                ss = line.split(" +");
                feature = Feature.valueOf(Integer.parseInt(ss[0]), ss[1], Integer.parseInt(ss[2]),
                        Double.parseDouble(ss[3]), Double.parseDouble(ss[4]));
                featureMap.put(ss[1], feature);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static SortedSet<Feature> vectorization(String text) throws Exception {
        SortedSet<Feature> result = Sets.newTreeSet();
        Multiset<String> doc = analysis(new StringReader(text));

        for (Multiset.Entry<String> word : doc.entrySet()) {
            if (!featureMap.containsKey(word.getElement())) {
                continue;
            }
            Feature feature = featureMap.get(word.getElement());
            feature.setWeight(weight(word.getCount(), feature.getDf(), N));
            result.add(feature);
        }
        LibSVMScale.scale(result, range);
        return result;
    }
}
