package com.aliyun.classifier.svm;

import java.text.DecimalFormat;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

import org.apache.commons.lang3.StringUtils;

import com.aliyun.classifier.Config;
import com.google.common.collect.Lists;
import com.google.common.math.DoubleMath;

public class LibSVMOneConfuseMatrix extends Config {

    private static final DecimalFormat df           = new DecimalFormat("0.######");
    private static final int[]         COLUMN_WITDH = new int[] { 30, 15, 15 };

    private List<Row>                  rows         = Lists.newArrayList();

    private List<Double>               accuracys    = Lists.newArrayList();

    public LibSVMOneConfuseMatrix() {
    }

    public void testSample(String category, List<String> lines, svm_model model) {
        double A = 0, B = lines.size();

        for (String line : lines) {
            StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
            double c = Double.parseDouble(st.nextToken());
            int m = st.countTokens() / 2;
            svm_node[] x = new svm_node[m];
            for (int j = 0; j < m; j++) {
                x[j] = new svm_node();
                x[j].index = Integer.parseInt(st.nextToken());
                x[j].value = Double.parseDouble(st.nextToken());
            }
            double pc = svm.svm_predict(model, x);

            if (c == pc)
                A++;
        }

        double accuracy = 0;
        if (B > 0) {
            accuracy = A / B;
        }
        this.accuracys.add(accuracy);
        this.rows.add(new Row(category, accuracy, (int) A, (int) B));
    }

    public List<String> getReport() {
        List<String> report = Lists.newArrayList();
        report.add("===============================================================================");
        report.add("CORPUS:" + CORPUS_NAME.toUpperCase());
        report.add("===============================================================================");
        StringBuilder header = new StringBuilder();
        header.append(StringUtils.rightPad("CATEGORY", COLUMN_WITDH[0]));
        header.append(StringUtils.rightPad("ACCURACY", COLUMN_WITDH[1]));
        header.append(StringUtils.rightPad("AB", COLUMN_WITDH[2]));
        report.add(header.toString());

        Collections.sort(rows);
        for (Row row : rows) {
            report.add(row.toString());
        }
        report.add(" ");
        report.add(StringUtils.rightPad("ACCURACY:", COLUMN_WITDH[0])
                + StringUtils.rightPad(df.format(DoubleMath.mean(this.accuracys)), COLUMN_WITDH[1]));
        return report;
    }

    private static class Row implements Comparable<Row> {
        public String category;
        public double accuracy;
        public int    A;
        public int    B;

        public Row(String category, double accuracy, int A, int B) {
            this.category = category;
            this.accuracy = accuracy;
            this.A = A;
            this.B = B;
        }

        @Override
        public String toString() {
            StringBuilder info = new StringBuilder(StringUtils.rightPad("[" + category.toUpperCase() + "]",
                    COLUMN_WITDH[0]));
            info.append(StringUtils.rightPad(df.format(accuracy), COLUMN_WITDH[1]));
            info.append(StringUtils.rightPad(this.A + " " + this.B, COLUMN_WITDH[2]));
            return info.toString();
        }

        @Override
        public int compareTo(Row o) {
            if (this.accuracy == o.accuracy) {
                return 0;
            }

            return this.accuracy > o.accuracy ? -1 : 1;
        }
    }
}
