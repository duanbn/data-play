package com.aliyun.classifier;

public class Feature implements Comparable<Feature> {

    private int    id;

    private String value;

    private int    df;

    private double weight;

    private double igScore;

    private double chiScore;

    private Feature() {
    }

    public static Feature valueOf(String value) {
        Feature feature = new Feature();
        feature.setValue(value);
        return feature;
    }

    public static Feature valueOf(int id) {
        Feature w = new Feature();
        w.id = id;
        return w;
    }

    public static Feature valueOf(int id, String value, int df, double igScore, double chiScore) {
        Feature feature = new Feature();
        feature.id = id;
        feature.value = value;
        feature.df = df;
        feature.igScore = igScore;
        feature.chiScore = chiScore;
        return feature;
    }

    @Override
    public int compareTo(Feature o) {
        return (int) (this.getId() - o.getId());
    }

    @Override
    public String toString() {
        return "Feature [id=" + id + ", value=" + value + ", df=" + df + ", weight=" + weight + ", igScore=" + igScore
                + ", chiScore=" + chiScore + "]";
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((value == null) ? 0 : value.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Feature other = (Feature) obj;
        if (value == null) {
            if (other.value != null)
                return false;
        } else if (!value.equals(other.value))
            return false;
        return true;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public int getDf() {
        return df;
    }

    public void setDf(int df) {
        this.df = df;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getIgScore() {
        return igScore;
    }

    public void setIgScore(double igScore) {
        this.igScore = igScore;
    }

    public double getChiScore() {
        return chiScore;
    }

    public void setChiScore(double chiScore) {
        this.chiScore = chiScore;
    }

}
