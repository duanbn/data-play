package com.aliyun.classifier.svm;

import java.io.FileReader;
import java.io.StringReader;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Word;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;

public class LibSVMTextVectorizer extends Config {

    private static final Map<String, Word> map = Maps.newHashMap();

    private static int                     N;

    static {
        try (FileReader fr = new FileReader(DICT)) {
            String[] ss = null;
            List<String> lines = IOUtils.readLines(fr);
            N = Integer.parseInt(lines.get(0));
            for (String line : lines.subList(1, lines.size())) {
                ss = line.split(" ");
                map.put(ss[1], Word.valueOf(Long.parseLong(ss[0]), ss[1], 0, Integer.parseInt(ss[2])));
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static List<Word> vectorization(String text) throws Exception {
        List<Word> result = Lists.newArrayList();
        Multiset<String> doc = analysis(new StringReader(text));
        for (Multiset.Entry<String> word : doc.entrySet()) {
            if (!map.containsKey(word.getElement())) {
                continue;
            }
            Word w = map.get(word.getElement());
            w.setTf(word.getCount());
            w.setScore(weight(w, N));
            result.add(w);
        }
        Collections.sort(result);
        return result;
    }

}
