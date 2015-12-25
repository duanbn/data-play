package com.aliyun.classifier;

public class Word implements Comparable<Word> {

    private long   id;

    private String value;

    private int    tf;

    private int    df;

    private double score;

    private double minScore;

    private double maxScore;

    private double quality;

    private Word() {
    }

    public static Word valueOf(String value) {
        Word w = new Word();
        w.setValue(value);
        return w;
    }

    public static Word valueOf(long id) {
        Word w = new Word();
        w.id = id;
        return w;
    }

    public static Word valueOf(long id, String value, int tf, int df) {
        Word w = new Word();
        w.id = id;
        w.value = value;
        w.tf = tf;
        w.df = df;
        return w;
    }

    @Override
    public int compareTo(Word o) {
        return (int) (this.getId() - o.getId());
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public int getTf() {
        return tf;
    }

    public void setTf(int tf) {
        this.tf = tf;
    }

    public int getDf() {
        return df;
    }

    public void setDf(int df) {
        this.df = df;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    public double getMinScore() {
        return minScore;
    }

    public void setMinScore(double minScore) {
        this.minScore = minScore;
    }

    public double getMaxScore() {
        return maxScore;
    }

    public void setMaxScore(double maxScore) {
        this.maxScore = maxScore;
    }

    public double getQuality() {
        return quality;
    }

    public void setQuality(double quality) {
        this.quality = quality;
    }

}
