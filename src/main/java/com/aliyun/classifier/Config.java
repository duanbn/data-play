package com.aliyun.classifier;

import java.io.File;
import java.io.FileReader;
import java.io.Reader;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.search.similarities.DefaultSimilarity;

import com.chenlb.mmseg4j.analysis.ComplexAnalyzer;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;

@SuppressWarnings("unchecked")
public class Config {

    protected static final ExecutorService      threadPool         = Executors.newFixedThreadPool(Runtime.getRuntime()
                                                                           .availableProcessors() * 5);

    protected static final double               DR_THRESHOLD;
    protected static final double               COST;
    protected static final double               GAMMA;

    protected static final String               CORPUS_NAME;
    protected static final File                 CORPUS_DB;
    protected static final File                 TRAIN_DB;

    protected static final File                 CATEGORYTF;
    protected static final File                 CHI;
    protected static final File                 DICT;
    protected static final File                 LABELINDEX;
    protected static final File                 REPORT;
    protected static final File                 RANGE;

    protected static final Analyzer             analyzer           = new ComplexAnalyzer();
    protected static final DefaultSimilarity    sim                = new DefaultSimilarity();

    protected static final Map<String, Integer> CATEGORY_NAME_CODE = Maps.newLinkedHashMap();
    protected static final Map<Integer, String> CATEGORY_CODE_NAME = Maps.newLinkedHashMap();

    protected static final Map<String, Double>  CATEGORY_PARAM     = Maps.newLinkedHashMap();

    static {
        TRAIN_DB = new File(System.getProperty("system.prop.basedir"), "train");
        if (!TRAIN_DB.exists()) {
            TRAIN_DB.mkdirs();
        }

        Configuration conf = null;
        try {
            conf = new PropertiesConfiguration("config.properties");
        } catch (Exception e) {
            throw new RuntimeException("load properties fail", e);
        }

        CORPUS_NAME = conf.getString("corpus.name");
        CORPUS_DB = new File(conf.getString("corpus.directory"));
        DR_THRESHOLD = conf.getDouble("corpus.dr.threshold");
        COST = conf.getDouble("corpus.cost");
        GAMMA = conf.getDouble("corpus.gamma");

        CATEGORYTF = new File(TRAIN_DB, CORPUS_NAME + ".tf");
        CHI = new File(TRAIN_DB, CORPUS_NAME + ".chi");
        DICT = new File(TRAIN_DB, CORPUS_NAME + ".dict");
        LABELINDEX = new File(TRAIN_DB, CORPUS_NAME + ".labelindex");
        REPORT = new File(TRAIN_DB, CORPUS_NAME + ".report");
        RANGE = new File(TRAIN_DB, CORPUS_NAME + ".range");

        // init category cost
        Iterator<String> keys = conf.getKeys("category");
        while (keys.hasNext()) {
            String key = keys.next();
            CATEGORY_PARAM.put(key.substring("category.".length()), conf.getDouble(key));
        }

        // init category
        int categoryId = 0;
        for (Map.Entry<String, Double> entry : CATEGORY_PARAM.entrySet()) {
            categoryId++;
            CATEGORY_CODE_NAME.put(categoryId, entry.getKey());
            CATEGORY_NAME_CODE.put(entry.getKey(), categoryId);
        }
    }

    protected static File getFile(String name, String prefix) {
        return new File(TRAIN_DB, name + "." + prefix);
    }

    protected static double weight(int tf, int df, int N, double score) {
        return sim.tf(tf) * sim.idf(df, N) * score;
    }

    protected static Multiset<String> analysis(File corpus) throws Exception {
        return analysis(new FileReader(corpus));
    }

    protected static Multiset<String> analysis(Reader reader) throws Exception {
        TokenStream tokenStream = analyzer.tokenStream("text", reader);
        CharTermAttribute termAtt = tokenStream.addAttribute(CharTermAttribute.class);
        tokenStream.reset();
        Multiset<String> words = ConcurrentHashMultiset.create();
        while (tokenStream.incrementToken()) {
            if (termAtt.length() > 0) {
                String word = tokenStream.getAttribute(CharTermAttribute.class).toString();
                words.add(word);
            }
        }
        tokenStream.end();
        tokenStream.close();

        return words;
    }

    public static class ConfigCategoryC {
        public double cost;

        public static ConfigCategoryC valueOf(double cost) {
            ConfigCategoryC instance = new ConfigCategoryC();
            instance.cost = cost;
            return instance;
        }
    }

}
