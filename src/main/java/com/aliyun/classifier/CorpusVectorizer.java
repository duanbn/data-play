package com.aliyun.classifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;

public class CorpusVectorizer extends Config {

    private int                                N                 = 0;
    private Map<String, Integer>               df                = Maps.newHashMap();
    private Map<String, Long>                  wordIdMap         = Maps.newHashMap();

    private Multimap<String, Multiset<String>> categoryTokenized = ArrayListMultimap.create();

    public void run() throws Exception {
        List<String> lines = Lists.newArrayList();
        for (Map.Entry<String, Integer> entry : CATEGORY_NAME_CODE.entrySet()) {
            lines.add(entry.getKey() + " " + entry.getValue());
        }
        try (FileWriter fw = new FileWriter(LABELINDEX)) {
            IOUtils.writeLines(lines, "\n", fw);
        }
        System.out.println("process category label index done");

        init();

        List<String> svmLines = Lists.newArrayList();
        List<String> svmtLines = Lists.newArrayList();
        for (final Map.Entry<String, Integer> entry : CATEGORY_NAME_CODE.entrySet()) {
            String category = entry.getKey();
            Integer categoryId = entry.getValue();

            Multimap<Integer, List<Feature>> svmFormats = processSvmFormat(category, categoryId);
            Map<Integer, List<List<Feature>>> svmDocsMap = Maps.newHashMap();
            Map<Integer, List<List<Feature>>> svmtDocsMap = Maps.newHashMap();
            split(svmDocsMap, svmtDocsMap, svmFormats, 0.6);

            List<String> oneSvmLines = build(svmLines, svmDocsMap);
            List<String> oneSvmtLines = build(svmtLines, svmtDocsMap);

            try (FileWriter fw = new FileWriter(getFile(category, "svm"))) {
                IOUtils.writeLines(oneSvmLines, "\n", fw);
            }
            try (FileWriter fw = new FileWriter(getFile(category, "svmt"))) {
                IOUtils.writeLines(oneSvmtLines, "\n", fw);
            }

            System.out.println("vectorization " + category + " done");
        }
        try (FileWriter fw = new FileWriter(getFile(CORPUS_NAME, "svm"))) {
            IOUtils.writeLines(svmLines, "\n", fw);
        }
        try (FileWriter fw = new FileWriter(getFile(CORPUS_NAME, "svmt"))) {
            IOUtils.writeLines(svmtLines, "\n", fw);
        }
    }

    private void init() throws Exception {
        Multiset<String> tokenized = null;
        for (Map.Entry<String, Integer> categoryEntry : CATEGORY_NAME_CODE.entrySet()) {
            for (File corpusFile : new File(CORPUS_DB, categoryEntry.getKey()).listFiles()) {
                tokenized = analysis(corpusFile);
                categoryTokenized.put(categoryEntry.getKey(), tokenized);
            }
        }

        try (BufferedReader br = new BufferedReader(new FileReader(DICT))) {
            String[] ss = null;
            List<String> lines = IOUtils.readLines(br);
            this.N = Integer.parseInt(lines.get(0));
            for (String line : lines.subList(1, lines.size())) {
                ss = line.split(" +");
                this.df.put(ss[1], Integer.parseInt(ss[2]));
                this.wordIdMap.put(ss[1], Long.parseLong(ss[0]));
            }
        }
    }

    private void split(Map<Integer, List<List<Feature>>> svmMap, Map<Integer, List<List<Feature>>> svmtMap,
                       Multimap<Integer, List<Feature>> svmFormats, double percentage) {
        for (Map.Entry<Integer, Collection<List<Feature>>> entry : svmFormats.asMap().entrySet()) {
            List<List<Feature>> docs = (List<List<Feature>>) entry.getValue();

            int plot = Double.valueOf(docs.size() * percentage).intValue();

            List<List<Feature>> svmDocs = svmMap.get(entry.getKey());
            if (svmDocs == null) {
                svmDocs = docs.subList(0, plot);
            } else {
                svmDocs.addAll(docs.subList(0, plot));
            }
            svmMap.put(entry.getKey(), svmDocs);

            List<List<Feature>> svmtDocs = svmtMap.get(entry.getKey());
            if (svmtDocs == null) {
                svmtDocs = docs.subList(plot, docs.size());
            } else {
                svmtDocs.addAll(docs.subList(plot, docs.size()));
            }
            svmtMap.put(entry.getKey(), svmtDocs);
        }
    }

    private List<String> build(List<String> lines, Map<Integer, List<List<Feature>>> docsMap) {
        List<String> oneLines = Lists.newArrayList();

        StringBuilder oneLine = null;
        StringBuilder line = null;
        for (Map.Entry<Integer, List<List<Feature>>> entry : docsMap.entrySet()) {

            for (List<Feature> doc : entry.getValue()) {

                if (doc.isEmpty()) {
                    continue;
                }

                oneLine = new StringBuilder();
                oneLine.append("1").append(" ");
                line = new StringBuilder();
                line.append(entry.getKey()).append(" ");

                for (Feature feature : doc) {
                    oneLine.append(feature.getId()).append(':').append(feature.getScore()).append(" ");
                    line.append(feature.getId()).append(':').append(feature.getScore()).append(" ");
                }

                oneLines.add(oneLine.toString());
                lines.add(line.toString());
            }

        }

        return oneLines;
    }

    private Multimap<Integer, List<Feature>> processSvmFormat(String category, int categoryId) throws Exception {
        // compute svm vector
        Multimap<Integer, List<Feature>> svmFormats = ArrayListMultimap.create();
        List<Feature> featuresVector = null;

        for (Multiset<String> doc : categoryTokenized.get(category)) {
            featuresVector = Lists.newArrayList();
            for (Multiset.Entry<String> word : doc.entrySet()) {
                if (!wordIdMap.containsKey(word.getElement()))
                    continue;
                Feature w = Feature.valueOf(wordIdMap.get(word.getElement()), word.getElement(), word.getCount(),
                        df.get(word.getElement()));
                w.setScore(weight(w, N));
                featuresVector.add(w);
            }
            Collections.sort(featuresVector);
            svmFormats.put(categoryId, featuresVector);
        }

        return svmFormats;
    }

}
