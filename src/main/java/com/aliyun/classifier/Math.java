package com.aliyun.classifier;

import com.google.common.math.DoubleMath;

public class Math {

    public static double variance(double[] values) {
        double diff = 0;
        double mean = DoubleMath.mean(values);
        for (int i = 0; i < values.length; i++) {
            diff += java.lang.Math.pow(values[i] - mean, 2);
        }
        return diff / values.length;
    }

    public static double sVariance(double[] values) {
        double variance = variance(values);
        return java.lang.Math.sqrt(variance);
    }
}
