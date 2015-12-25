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
import com.aliyun.classifier.svm.LibSVMScale.Range;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;

public class LibSVMTrain extends Config {

    private long                               maxWordId         = 0;
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

        final LibSVMConfuseMatrix cMatrix = new LibSVMConfuseMatrix();

        final LibSVMTrain main = new LibSVMTrain();
        main.init();
        for (final Map.Entry<String, Integer> entry : CATEGORY_NAME_CODE.entrySet()) {
            if (CATEGORY_PARAM.containsKey(entry.getKey()) && CATEGORY_PARAM.get(entry.getKey()).isTrain) {
                main.doTrain(entry.getKey(), entry.getValue(), cMatrix);
            }
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
                ss = line.split(" ");
                this.df.put(ss[1], Integer.parseInt(ss[2]));
                this.wordIdMap.put(ss[1], Long.parseLong(ss[0]));
                this.maxWordId = Long.parseLong(ss[0]);
            }
        }
    }

    public void doTrain(String category, int categoryId, LibSVMConfuseMatrix cMatrix) throws Exception {
        Multimap<Integer, List<Word>> svmFormats = processSvmFormat(category, categoryId);

        Map<Integer, List<List<Word>>> svmDocsMap = Maps.newHashMap();
        Map<Integer, List<List<Word>>> svmtDocsMap = Maps.newHashMap();

        split(svmDocsMap, svmtDocsMap, svmFormats, 0.6);
        //        scale(svmDocsMap, svmtDocsMap, category);

        List<String> svmLines = Lists.newArrayList();
        build(svmLines, svmDocsMap);
        try (FileWriter fw = new FileWriter(getFile(category, "svm"))) {
            IOUtils.writeLines(svmLines, "\n", fw);
        }
        List<String> svmtLines = Lists.newArrayList();
        build(svmtLines, svmtDocsMap);
        try (FileWriter fw = new FileWriter(getFile(category, "svmt"))) {
            IOUtils.writeLines(svmtLines, "\n", fw);
        }

        // train model
        svm_parameter param = getSvm_parameter(categoryId);
        svm_problem prob = getSvm_problem(svmLines, param);
        svm_model model = svm.svm_train(prob, param);
        svm.svm_save_model(getFile(category, "model").getAbsolutePath(), model);

        cMatrix.testSample(svmtLines, category, model, param);
    }

    private svm_parameter getSvm_parameter(int categoryId) {
        svm_parameter param = new svm_parameter();

        if (IS_ONECLASS) {
            param.svm_type = svm_parameter.ONE_CLASS;
            param.nr_weight = 0;
            param.weight_label = new int[0];
            param.weight = new double[0];
            param.nu = 0.99;
        } else {
            param.svm_type = svm_parameter.C_SVC;
            // compute punish factor
            int[] weightLabel = null;
            double[] weight = null;
            weightLabel = new int[] { categoryId };
            weight = new double[] { getParameterC(categoryId) };
            param.nr_weight = 1;
            param.weight_label = weightLabel;
            param.weight = weight;
            param.nu = 0.1;
        }

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

    @Deprecated
    private void scale(Map<Integer, List<List<Word>>> svmMap, Map<Integer, List<List<Word>>> svmtMap, String category) {
        List<Word> features = Lists.newArrayList();
        for (List<List<Word>> a : svmMap.values()) {
            for (List<Word> b : a) {
                features.addAll(b);
            }
        }
        Range range = LibSVMScale.range(features, maxWordId);
        LibSVMScale.serialize(range, category);
        for (List<List<Word>> a : svmMap.values()) {
            for (List<Word> b : a) {
                LibSVMScale.scale(b, range);
            }
        }
        for (List<List<Word>> a : svmtMap.values()) {
            for (List<Word> b : a) {
                LibSVMScale.scale(b, range);
            }
        }
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

        if (IS_ONECLASS) {
            for (Multiset<String> doc : categoryTokenized.get(category)) {
                int labelIndex = 1;
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
                svmFormats.put(labelIndex, featuresVector);
            }
        } else {
            for (Map.Entry<String, Collection<Multiset<String>>> entry : categoryTokenized.asMap().entrySet()) {

                int labelIndex = -1;
                if (entry.getKey().equals(category)) {
                    labelIndex = categoryId;
                }

                for (Multiset<String> doc : entry.getValue()) {
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
                    svmFormats.put(labelIndex, featuresVector);
                }
            }
        }

        return svmFormats;
    }

}
