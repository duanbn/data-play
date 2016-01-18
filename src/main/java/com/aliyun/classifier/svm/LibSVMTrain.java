package com.aliyun.classifier.svm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.Vector;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Word;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;

public class LibSVMTrain extends Config {

    private int                                N                 = 0;
    private Map<String, Integer>               df                = Maps.newHashMap();
    private Map<String, Long>                  wordIdMap         = Maps.newHashMap();

    private Multimap<String, Multiset<String>> categoryTokenized = ArrayListMultimap.create();

    public void vectorization() throws Exception {
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

            Multimap<Integer, List<Word>> svmFormats = processSvmFormat(category, categoryId);
            Map<Integer, List<List<Word>>> svmDocsMap = Maps.newHashMap();
            Map<Integer, List<List<Word>>> svmtDocsMap = Maps.newHashMap();
            split(svmDocsMap, svmtDocsMap, svmFormats, 0.6);

            build(svmLines, svmDocsMap);
            build(svmtLines, svmtDocsMap);

            System.out.println("vectorization " + category + " done");
        }
        try (FileWriter fw = new FileWriter(getFile(CORPUS_NAME, "svm"))) {
            IOUtils.writeLines(svmLines, "\n", fw);
        }
        try (FileWriter fw = new FileWriter(getFile(CORPUS_NAME, "svmt"))) {
            IOUtils.writeLines(svmtLines, "\n", fw);
        }
    }

    public void train() throws Exception {
        final LibSVMConfuseMatrix cMatrix = new LibSVMConfuseMatrix();

        svm_parameter param = getSvm_parameter();
        svm_model model = null;
        try (FileReader svmFr = new FileReader(getFile(CORPUS_NAME, "svm"))) {
            List<String> svmLines = IOUtils.readLines(svmFr);

            svm_problem prob = getSvm_problem(svmLines, param);
            model = svm.svm_train(prob, param);
            svm.svm_save_model(getFile(CORPUS_NAME, "model").getAbsolutePath(), model);
        }

        try (FileReader svmtFr = new FileReader(getFile(CORPUS_NAME, "svmt"))) {
            List<String> svmtLines = IOUtils.readLines(svmtFr);
            cMatrix.testSample(svmtLines, model);
        }

        try (FileWriter fw = new FileWriter(REPORT, true)) {
            IOUtils.writeLines(cMatrix.getReport(), "\n", fw);
        }

        System.out.println("done");
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

    private svm_parameter getSvm_parameter() {
        svm_parameter param = new svm_parameter();

        param.svm_type = svm_parameter.C_SVC;
        // compute punish factor
        int[] weightLabel = new int[CATEGORY_PARAM.size()];
        double[] weight = new double[CATEGORY_PARAM.size()];
        int i = 0;
        for (Map.Entry<String, Double> entry : CATEGORY_PARAM.entrySet()) {
            weightLabel[i] = CATEGORY_NAME_CODE.get(entry.getKey());
            weight[i] = entry.getValue();
            i++;
        }
        param.nr_weight = CATEGORY_PARAM.size();
        param.weight_label = weightLabel;
        param.weight = weight;
        param.nu = 0.1;

        param.kernel_type = svm_parameter.RBF;
        param.degree = 3;
        param.gamma = 0; // 1/num_features
        param.coef0 = 0;
        param.cache_size = 200;
        param.C = 1;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;

        return param;
    }

    private svm_problem getSvm_problem(List<String> lines, svm_parameter param) {
        Vector<Double> vy = new Vector<Double>();
        Vector<svm_node[]> vx = new Vector<svm_node[]>();
        int max_index = 0;

        for (String line : lines) {
            StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
            vy.addElement(Double.parseDouble(st.nextToken()));
            int m = st.countTokens() / 2;
            svm_node[] x = new svm_node[m];
            for (int j = 0; j < m; j++) {
                x[j] = new svm_node();
                x[j].index = Integer.parseInt(st.nextToken());
                x[j].value = Double.parseDouble(st.nextToken());
            }
            if (m > 0)
                max_index = Math.max(max_index, x[m - 1].index);
            vx.addElement(x);
        }

        svm_problem prob = new svm_problem();
        prob.l = vy.size();
        prob.x = new svm_node[prob.l][];
        for (int i = 0; i < prob.l; i++)
            prob.x[i] = vx.elementAt(i);
        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++)
            prob.y[i] = vy.elementAt(i);

        if (param.gamma == 0 && max_index > 0)
            param.gamma = 1.0 / max_index;

        return prob;
    }

    private void split(Map<Integer, List<List<Word>>> svmMap, Map<Integer, List<List<Word>>> svmtMap,
                       Multimap<Integer, List<Word>> svmFormats, double percentage) {
        for (Map.Entry<Integer, Collection<List<Word>>> entry : svmFormats.asMap().entrySet()) {
            List<List<Word>> docs = (List<List<Word>>) entry.getValue();

            int plot = Double.valueOf(docs.size() * percentage).intValue();

            List<List<Word>> svmDocs = svmMap.get(entry.getKey());
            if (svmDocs == null) {
                svmDocs = docs.subList(0, plot);
            } else {
                svmDocs.addAll(docs.subList(0, plot));
            }
            svmMap.put(entry.getKey(), svmDocs);

            List<List<Word>> svmtDocs = svmtMap.get(entry.getKey());
            if (svmtDocs == null) {
                svmtDocs = docs.subList(plot, docs.size());
            } else {
                svmtDocs.addAll(docs.subList(plot, docs.size()));
            }
            svmtMap.put(entry.getKey(), svmtDocs);
        }
    }

    private void build(List<String> lines, Map<Integer, List<List<Word>>> docsMap) {
        StringBuilder line = null;
        for (Map.Entry<Integer, List<List<Word>>> entry : docsMap.entrySet()) {
            for (List<Word> doc : entry.getValue()) {
                if (doc.isEmpty()) {
                    continue;
                }
                line = new StringBuilder();
                line.append(entry.getKey()).append(" ");
                for (Word feature : doc) {
                    line.append(feature.getId()).append(':').append(feature.getScore()).append(" ");
                }
                lines.add(line.toString());
            }
        }
    }

    private Multimap<Integer, List<Word>> processSvmFormat(String category, int categoryId) throws Exception {
        // compute svm vector
        Multimap<Integer, List<Word>> svmFormats = ArrayListMultimap.create();
        List<Word> featuresVector = null;

        for (Multiset<String> doc : categoryTokenized.get(category)) {
            featuresVector = Lists.newArrayList();
            for (Multiset.Entry<String> word : doc.entrySet()) {
                if (!wordIdMap.containsKey(word.getElement()))
                    continue;
                Word w = Word.valueOf(wordIdMap.get(word.getElement()), word.getElement(), word.getCount(),
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
