package com.aliyun.classifier.svm;

import java.io.FileReader;
import java.io.FileWriter;
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

public class LibSVMTrain extends Config {

    public void run() throws Exception {
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

}
