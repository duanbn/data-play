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

/**
 * 非线程安全类
 * 
 * @author shanwei Jan 26, 2016 10:12:40 AM
 */
public class LibSVMConfuseMatrix extends Config {

    private static final DecimalFormat dformat      = new DecimalFormat("0.######");
    private static final int[]         COLUMN_WITDH = new int[] { 30, 20, 20, 20, 20 };

    private double                     testCount;
    private double                     accuracyCount;
    private int[][]                    maxtirx      = new int[CATEGORY_PARAM.size() + 1][3];
    private List<Row>                  rows         = Lists.newArrayList();

    private List<Double>               recalls      = Lists.newArrayList();
    private List<Double>               precisions   = Lists.newArrayList();
    private List<Double>               fscores      = Lists.newArrayList();

    public LibSVMConfuseMatrix() {
    }

    public void testSample(List<String> lines, svm_model model) {
        testCount += lines.size();

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

            if (c == pc) {
                maxtirx[new Double(c).intValue()][0] += 1;
                this.accuracyCount++;
            } else {
                maxtirx[new Double(c).intValue()][1] += 1;
                maxtirx[new Double(pc).intValue()][2] += 1;
            }
        }

        for (int i = 1; i < maxtirx.length; i++) {
            int A = maxtirx[i][0];
            int B = maxtirx[i][1];
            int C = maxtirx[i][2];
            double recall = recall(A, B) * 100;
            this.recalls.add(recall);
            double precision = precision(A, C) * 100;
            this.precisions.add(precision);
            double fscore = fscore(A, B, C);
            this.fscores.add(fscore);

            this.rows.add(new Row(CATEGORY_CODE_NAME.get(i), recall, precision, fscore, (int) A, (int) B, (int) C));
        }
    }

    public List<String> getReport() {
        List<String> report = Lists.newArrayList();
        report.add("===============================================================================");
        report.add("CORPUS:" + CORPUS_NAME.toUpperCase());
        report.add("===============================================================================");
        StringBuilder header = new StringBuilder();
        header.append(StringUtils.rightPad("CATEGORY", COLUMN_WITDH[0]));
        header.append(StringUtils.rightPad("RECALL", COLUMN_WITDH[1]));
        header.append(StringUtils.rightPad("PRECISION", COLUMN_WITDH[2]));
        header.append(StringUtils.rightPad("FSCORE", COLUMN_WITDH[3]));
        header.append(StringUtils.rightPad("ABC", COLUMN_WITDH[4]));
        report.add(header.toString());

        Collections.sort(rows);
        for (Row row : rows) {
            report.add(row.toString());
        }
        report.add("---------------------------------------------");
        report.add(StringUtils.rightPad("ACCURACY:", COLUMN_WITDH[0])
                + StringUtils.rightPad(String.valueOf(accuracyCount / testCount), COLUMN_WITDH[1])
                + StringUtils.rightPad("total:" + this.testCount, COLUMN_WITDH[2])
                + StringUtils.rightPad("true:" + this.accuracyCount, COLUMN_WITDH[3])
                + StringUtils.rightPad("false:" + (this.testCount - this.accuracyCount), COLUMN_WITDH[4]));
        report.add(StringUtils.rightPad("MEAN-RECALL:", COLUMN_WITDH[0])
                + StringUtils.rightPad(dformat.format(DoubleMath.mean(recalls)), COLUMN_WITDH[1]));
        report.add(StringUtils.rightPad("MEAN-PRECISION:", COLUMN_WITDH[0])
                + StringUtils.rightPad(dformat.format(DoubleMath.mean(precisions)), COLUMN_WITDH[1]));
        report.add(StringUtils.rightPad("MEAN-FSCORE:", COLUMN_WITDH[0])
                + StringUtils.rightPad(dformat.format(DoubleMath.mean(fscores)), COLUMN_WITDH[1]));
        report.add(StringUtils.rightPad("COST:", COLUMN_WITDH[0])
                + StringUtils.rightPad(String.valueOf(COST), COLUMN_WITDH[1]));
        report.add(StringUtils.rightPad("GAMMA:", COLUMN_WITDH[0])
                + StringUtils.rightPad(String.valueOf(GAMMA), COLUMN_WITDH[1]));
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

    private static class Row implements Comparable<Row> {
        public String category;
        public double recall;
        public double precision;
        public double fscore;
        public int    A;
        public int    B;
        public int    C;

        public Row(String category, double recall, double precision, double fscore, int A, int B, int C) {
            this.category = category;
            this.recall = recall;
            this.precision = precision;
            this.fscore = fscore;
            this.A = A;
            this.B = B;
            this.C = C;
        }

        @Override
        public String toString() {
            StringBuilder info = new StringBuilder(StringUtils.rightPad("[" + category.toLowerCase() + "]",
                    COLUMN_WITDH[0]));
            info.append(StringUtils.rightPad(dformat.format(recall), COLUMN_WITDH[1]));
            info.append(StringUtils.rightPad(dformat.format(precision), COLUMN_WITDH[2]));
            info.append(StringUtils.rightPad(dformat.format(fscore), COLUMN_WITDH[3]));
            info.append(StringUtils.rightPad(this.A + " " + this.B + " " + this.C, COLUMN_WITDH[4]));
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
