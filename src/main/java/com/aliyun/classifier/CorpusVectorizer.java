package com.aliyun.classifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;

import org.apache.commons.io.IOUtils;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;
import com.google.common.collect.Sets;

public class CorpusVectorizer extends Config {

    private int                   N           = 0;
    private Map<String, Integer>  wordIdMap   = Maps.newHashMap();
    private Map<Integer, Integer> df          = Maps.newHashMap();
    private Map<Integer, Double>  igScoreMap  = Maps.newHashMap();
    private Map<Integer, Double>  chiScoreMap = Maps.newHashMap();
    private Map<Integer, Double>  scoreMap    = Maps.newHashMap();

    public CorpusVectorizer() throws Exception {
        try (BufferedReader br = new BufferedReader(new FileReader(DICT))) {
            String[] ss = null;
            List<String> lines = IOUtils.readLines(br);
            ss = lines.get(0).split(",");
            this.N = Integer.parseInt(ss[0]);
            for (String line : lines.subList(1, lines.size())) {
                ss = line.split(" +");
                int featureId = Integer.parseInt(ss[0]);
                this.wordIdMap.put(ss[1], featureId);
                this.df.put(featureId, Integer.parseInt(ss[2]));
                this.igScoreMap.put(featureId, Double.parseDouble(ss[3]));
                this.chiScoreMap.put(featureId, Double.parseDouble(ss[4]));
                this.scoreMap.put(featureId, Double.parseDouble(ss[5]));
            }
        }
        System.out.println("load " + DICT.getName() + " done");
    }

    public void run() throws Exception {
        Multimap<Integer, SortedSet<Feature>> categoryFeatures = ArrayListMultimap.create();
        for (Map.Entry<String, Integer> entry : CATEGORY_NAME_CODE.entrySet()) {
            for (File corpusFile : new File(CORPUS_DB, entry.getKey()).listFiles()) {

                Multiset<String> tokenized = analysis(corpusFile);
                double max = Double.MIN_VALUE, min = Double.MAX_VALUE;
                SortedSet<Feature> features = Sets.newTreeSet();
                for (Multiset.Entry<String> token : tokenized.entrySet()) {
                    if (wordIdMap.containsKey(token.getElement())) {
                        String value = token.getElement();
                        int featureId = wordIdMap.get(value);
                        int tf = token.getCount();
                        int df = this.df.get(featureId);
                        double score = this.scoreMap.get(featureId);
                        double igScore = this.igScoreMap.get(featureId);
                        double chiScore = this.chiScoreMap.get(featureId);

                        double weight = weight(tf, df, N, score);
                        Feature feature = Feature.valueOf(featureId, value, df, igScore, chiScore, score);
                        feature.setWeight(weight);
                        features.add(feature);

                        max = java.lang.Math.max(weight, max);
                        min = java.lang.Math.min(weight, min);
                    }
                }
                for (Feature feature : features) {
                    if (max == min) {
                        continue;
                    }
                    double norm = (feature.getWeight() - min) / (max - min);
                    feature.setWeight(norm);
                }
                categoryFeatures.put(entry.getValue(), features);
            }
        }

        List<String> svmLines = Lists.newArrayList();
        List<String> svmtLines = Lists.newArrayList();
        for (final Map.Entry<String, Integer> entry : CATEGORY_NAME_CODE.entrySet()) {
            //            String category = entry.getKey();
            Integer categoryId = entry.getValue();

            Multimap<Integer, SortedSet<Feature>> svmFormats = ArrayListMultimap.create();
            for (SortedSet<Feature> features : categoryFeatures.get(categoryId)) {
                svmFormats.put(categoryId, features);
            }
            Map<Integer, List<SortedSet<Feature>>> svmDocsMap = Maps.newHashMap();
            Map<Integer, List<SortedSet<Feature>>> svmtDocsMap = Maps.newHashMap();
            split(svmDocsMap, svmtDocsMap, svmFormats, 0.6);
            build(svmLines, svmDocsMap);
            build(svmtLines, svmtDocsMap);

            //            List<String> oneSvmLines = build(svmLines, svmDocsMap);
            //            List<String> oneSvmtLines = build(svmtLines, svmtDocsMap);

            //            try (FileWriter fw = new FileWriter(getFile(category, "svm"))) {
            //                IOUtils.writeLines(oneSvmLines, "\n", fw);
            //            }
            //            try (FileWriter fw = new FileWriter(getFile(category, "svmt"))) {
            //                IOUtils.writeLines(oneSvmtLines, "\n", fw);
            //            }

            //            System.out.println("vectorization " + category + " done");
        }
        try (FileWriter fw = new FileWriter(getFile(CORPUS_NAME, "svm"))) {
            IOUtils.writeLines(svmLines, "\n", fw);
        }
        try (FileWriter fw = new FileWriter(getFile(CORPUS_NAME, "svmt"))) {
            IOUtils.writeLines(svmtLines, "\n", fw);
        }

        System.out.println("vectorization all feature done");
    }

    private void split(Map<Integer, List<SortedSet<Feature>>> svmMap, Map<Integer, List<SortedSet<Feature>>> svmtMap,
                       Multimap<Integer, SortedSet<Feature>> svmFormats, double percentage) {
        for (Map.Entry<Integer, Collection<SortedSet<Feature>>> entry : svmFormats.asMap().entrySet()) {
            List<SortedSet<Feature>> docs = (List<SortedSet<Feature>>) entry.getValue();

            int plot = Double.valueOf(docs.size() * percentage).intValue();

            List<SortedSet<Feature>> svmDocs = svmMap.get(entry.getKey());
            if (svmDocs == null) {
                svmDocs = docs.subList(0, plot);
            } else {
                svmDocs.addAll(docs.subList(0, plot));
            }
            svmMap.put(entry.getKey(), svmDocs);

            List<SortedSet<Feature>> svmtDocs = svmtMap.get(entry.getKey());
            if (svmtDocs == null) {
                svmtDocs = docs.subList(plot, docs.size());
            } else {
                svmtDocs.addAll(docs.subList(plot, docs.size()));
            }
            svmtMap.put(entry.getKey(), svmtDocs);
        }
    }

    private List<String> build(List<String> lines, Map<Integer, List<SortedSet<Feature>>> docsMap) {
        List<String> oneLines = Lists.newArrayList();

        StringBuilder oneLine = null;
        StringBuilder line = null;
        for (Map.Entry<Integer, List<SortedSet<Feature>>> entry : docsMap.entrySet()) {

            for (SortedSet<Feature> doc : entry.getValue()) {

                if (doc.isEmpty()) {
                    continue;
                }

                oneLine = new StringBuilder();
                oneLine.append("1").append(" ");
                line = new StringBuilder();
                line.append(entry.getKey()).append(" ");

                for (Feature feature : doc) {
                    oneLine.append(feature.getId()).append(':').append(feature.getWeight()).append(" ");
                    line.append(feature.getId()).append(':').append(feature.getWeight()).append(" ");
                }

                oneLines.add(oneLine.toString());
                lines.add(line.toString());
            }

        }

        return oneLines;
    }

}
