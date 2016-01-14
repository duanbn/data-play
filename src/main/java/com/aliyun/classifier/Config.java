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

    protected static final ExecutorService              threadPool         = Executors
                                                                                   .newFixedThreadPool(Runtime
                                                                                           .getRuntime()
                                                                                           .availableProcessors() * 5);

    protected static String                             CORPUS_NAME;
    protected static File                               CORPUS_DB;
    protected static File                               TRAIN_DB;

    protected static File                               CHI;
    protected static File                               DICT;
    protected static File                               LABELINDEX;
    protected static File                               REPORT;

    protected static final Analyzer                     analyzer           = new ComplexAnalyzer();
    protected static final DefaultSimilarity            sim                = new DefaultSimilarity();

    protected static final Map<String, Integer>         CATEGORY_NAME_CODE = Maps.newTreeMap();
    protected static final Map<Integer, String>         CATEGORY_CODE_NAME = Maps.newTreeMap();

    protected static final Map<String, ConfigCategoryC> CATEGORY_PARAM     = Maps.newLinkedHashMap();

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
        CHI = new File(TRAIN_DB, CORPUS_NAME + ".chi");
        DICT = new File(TRAIN_DB, CORPUS_NAME + ".dict");
        LABELINDEX = new File(TRAIN_DB, CORPUS_NAME + ".labelindex");
        REPORT = new File(TRAIN_DB, CORPUS_NAME + ".report");
        CORPUS_DB = new File(conf.getString("corpus.directory"));

        // init category cost
        Iterator<String> keys = conf.getKeys("category");
        while (keys.hasNext()) {
            String key = keys.next();
            String[] ss = conf.getString(key).split(" ");
            CATEGORY_PARAM.put(key.substring("category.".length()),
                    ConfigCategoryC.valueOf(Double.parseDouble(ss[0]), Boolean.parseBoolean(ss[1])));
        }

        // init category
        int categoryId = 0;
        for (String category : CATEGORY_PARAM.keySet()) {
            categoryId++;
            CATEGORY_CODE_NAME.put(categoryId, category);
            CATEGORY_NAME_CODE.put(category, categoryId);
        }
    }

    protected static File getFile(String name, String prefix) {
        return new File(TRAIN_DB, name + "." + prefix);
    }

    protected static double weight(Word word, int N) {
        return sim.tf(word.getTf()) * sim.idf(word.getDf(), N);
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

    protected static double getParameterC(int categoryId) {
        String category = CATEGORY_CODE_NAME.get(categoryId);

        ConfigCategoryC param = CATEGORY_PARAM.get(category);
        if (param != null) {
            return param.cost;
        } else {
            return 1.0;
        }
    }

    public static class ConfigCategoryC {
        public double  cost;
        public boolean isTrain;

        public static ConfigCategoryC valueOf(double cost, boolean isTrain) {
            ConfigCategoryC instance = new ConfigCategoryC();
            instance.cost = cost;
            instance.isTrain = isTrain;
            return instance;
        }
    }

}
