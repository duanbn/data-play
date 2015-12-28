package com.aliyun.classifier.svm;

import java.io.FileWriter;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.io.IOUtils;

import com.aliyun.classifier.Config;
import com.aliyun.classifier.Word;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;

public class LibSVMDf extends Config {

    public void run(int N, Multimap<String, Set<String>> categoryTokenized, List<Word> tf) throws Exception {

        Map<String, Integer> df = Maps.newHashMap();

        // compute df
        long start = System.currentTimeMillis();
        List<String> lines = Lists.newArrayList(String.valueOf(N));
        long wordId = 0;
        StringBuilder line = null;
        for (Word word : tf) {
            int docFreq = 0;
            for (Set<String> docTokenized : categoryTokenized.values()) {
                if (docTokenized.contains(word.getValue())) {
                    docFreq++;
                }
            }
            if (docFreq > N * 0.9) {
                continue;
            }
            df.put(word.getValue(), docFreq);

            wordId++;

            line = new StringBuilder();
            line.append(wordId).append(" ");
            line.append(word.getValue()).append(" ");
            line.append(df.get(word.getValue())).append(" ");
            line.append(word.getQuality());
            lines.add(line.toString());
        }

        try (FileWriter fw = new FileWriter(DICT)) {
            IOUtils.writeLines(lines, "\n", fw);
        }

        System.out.println("compute df done " + (System.currentTimeMillis() - start) + "ms");

    }

}
