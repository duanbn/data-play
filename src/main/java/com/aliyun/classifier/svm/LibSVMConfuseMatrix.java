package com.aliyun.classifier.svm;

import java.text.DecimalFormat;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;

import org.apache.commons.lang3.StringUtils;

import com.aliyun.classifier.Config;
import com.google.common.collect.Lists;
import com.google.common.math.DoubleMath;

public class LibSVMConfuseMatrix extends Config {

    private static final DecimalFormat df           = new DecimalFormat("0.######");
    private static final int[]         COLUMN_WITDH = new int[] { 20, 15, 15, 15, 10, 20 };

    private List<Row>                  rows         = Lists.newArrayList();

    private List<Double>               recalls      = Lists.newArrayList();
    private List<Double>               precisions   = Lists.newArrayList();
    private List<Double>               fscores      = Lists.newArrayList();

    public LibSVMConfuseMatrix() {
    }

    public void testSample(List<String> lines, String category, svm_model model, svm_parameter svm_param) {
        double A = 0, B = 0, C = 0, D = 0;

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

            if (c != -1) {
                if (pc == c) {
                    A++;
                } else {
                    B++;
                }
            } else {
                if (pc == c) {
                    D++;
                } else {
                    C++;
                }
            }
        }

        double recall = recall(A, B) * 100;
        this.recalls.add(recall);
        double precision = precision(A, C) * 100;
        this.precisions.add(precision);
        double fscore = fscore(A, B, C);
        this.fscores.add(fscore);

        this.rows.add(new Row(category, recall, precision, fscore, (int) A, (int) B, (int) C, (int) D));
    }

    public List<String> getReport() {
        List<String> report = Lists.newArrayList();
        StringBuilder header = new StringBuilder();
        header.append(StringUtils.rightPad("CATEGORY", COLUMN_WITDH[0]));
        header.append(StringUtils.rightPad("RECALL", COLUMN_WITDH[1]));
        header.append(StringUtils.rightPad("PRECISION", COLUMN_WITDH[2]));
        header.append(StringUtils.rightPad("FSCORE", COLUMN_WITDH[3]));
        header.append(StringUtils.rightPad("COST", COLUMN_WITDH[4]));
        header.append(StringUtils.rightPad("ABCD", COLUMN_WITDH[5]));
        report.add("===============================================================================");
        report.add("CORPUS:" + CORPUS_NAME.toUpperCase());
        report.add("===============================================================================");
        report.add(header.toString());

        Collections.sort(rows);
        for (Row row : rows) {
            report.add(row.toString());
        }
        report.add("\n");
        report.add(StringUtils.rightPad("RECALL:", COLUMN_WITDH[0])
                + StringUtils.rightPad(df.format(DoubleMath.mean(recalls)), COLUMN_WITDH[1]));
        report.add(StringUtils.rightPad("PRECISION:", COLUMN_WITDH[0])
                + StringUtils.rightPad(df.format(DoubleMath.mean(precisions)), COLUMN_WITDH[1]));
        report.add(StringUtils.rightPad("FSCORE:", COLUMN_WITDH[0])
                + StringUtils.rightPad(df.format(DoubleMath.mean(fscores)), COLUMN_WITDH[1]));
        return report;
    }

    public double fscore(double A, double B, double C) {
        double recall = recall(A, B);
        double precision = precision(A, C);

        if (recall == 0) {
            return 0;
        }
        if (precision == 0) {
            return 0;
        }
        return 2 * precision * recall / (recall + precision);
    }

    public double recall(double A, double B) {
        if (A == 0) {
            return 0;
        }
        return A / (A + B);
    }

    public double precision(double A, double C) {
        if (A + C == 0) {
            return 0;
        }
        return A / (A + C);
    }

    public double accuracy(double A, double B, double C, double D) {
        return (A + D) / (A + B + C + D);
    }

    private static class Row implements Comparable<Row> {
        public String category;
        public double recall;
        public double precision;
        public double fscore;
        public int    A;
        public int    B;
        public int    C;
        public int    D;

        public Row(String category, double recall, double precision, double fscore, int A, int B, int C, int D) {
            this.category = category;
            this.recall = recall;
            this.precision = precision;
            this.fscore = fscore;
            this.A = A;
            this.B = B;
            this.C = C;
            this.D = D;
        }

        @Override
        public String toString() {
            StringBuilder info = new StringBuilder(StringUtils.rightPad("[" + category.toUpperCase() + "]",
                    COLUMN_WITDH[0]));
            info.append(StringUtils.rightPad(df.format(recall), COLUMN_WITDH[1]));
            info.append(StringUtils.rightPad(df.format(precision), COLUMN_WITDH[2]));
            info.append(StringUtils.rightPad(df.format(fscore), COLUMN_WITDH[3]));
            info.append(StringUtils.rightPad(String.valueOf(CATEGORY_PARAM.get(category).cost), COLUMN_WITDH[4]));
            info.append(StringUtils.rightPad(this.A + " " + this.B + " " + this.C + " " + this.D, COLUMN_WITDH[5]));
            return info.toString();
        }

        @Override
        public int compareTo(Row o) {
            if (this.fscore == o.fscore) {
                return 0;
            }

            return this.fscore > o.fscore ? -1 : 1;
        }
    }
}
