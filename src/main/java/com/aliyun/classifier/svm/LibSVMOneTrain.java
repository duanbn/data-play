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

public class LibSVMOneTrain extends Config {

    public void run() throws Exception {
        LibSVMOneConfuseMatrix cMatrix = new LibSVMOneConfuseMatrix();

        svm_model model = null;
        for (Map.Entry<String, Double> entry : CATEGORY_PARAM.entrySet()) {
            try (FileReader svmFr = new FileReader(getFile(entry.getKey(), "svm"));
                    FileReader svmtFr = new FileReader(getFile(entry.getKey(), "svmt"))) {
                List<String> svmLines = IOUtils.readLines(svmFr);

                svm_parameter param = getSvm_parameter(entry.getValue());
                svm_problem prob = getSvm_problem(svmLines, param);
                model = svm.svm_train(prob, param);
                svm.svm_save_model(getFile(entry.getKey(), "model").getAbsolutePath(), model);

                List<String> svmtLines = IOUtils.readLines(svmtFr);
                cMatrix.testSample(entry.getKey(), svmtLines, model);
            }
        }

        try (FileWriter fw = new FileWriter(REPORT, true)) {
            IOUtils.writeLines(cMatrix.getReport(), "\n", fw);
        }

        System.out.println("done");
    }

    private svm_parameter getSvm_parameter(double nu) {
        svm_parameter param = new svm_parameter();

        param.svm_type = svm_parameter.ONE_CLASS;
        // compute punish factor
        int[] weightLabel = new int[0];
        double[] weight = new double[0];
        param.nr_weight = 0;
        param.weight_label = weightLabel;
        param.weight = weight;
        param.nu = nu;

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
