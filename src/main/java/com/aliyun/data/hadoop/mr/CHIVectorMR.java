package com.aliyun.data.hadoop.mr;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.VectorHelper;

import com.google.common.collect.Maps;

public class CHIVectorMR extends AbstractJob {

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new Configuration(), new CHIVectorMR(), args);
    }

    @Override
    public int run(String[] args) throws Exception {
        addInputOption();
        addOption("dictionary", "d", "The dictionary file.", false);
        addOption("dictionaryType", "dt", "The dictionary file type (text|seqfile)", false);

        if (parseArguments(args, false, true) == null) {
            return -1;
        }

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path input = getInputPath();

        Path chiPath = new Path(input.getParent(), "CHI");
        Map<String, Double> chiMap = loadCHI(fs, chiPath);

        Path tfidfVectorPath = new Path(input, "tfidf-vectors");

        FileStatus fileStatus = fs.getFileStatus(tfidfVectorPath);
        Path[] pathArr;
        if (fileStatus.isDirectory()) {
            pathArr = FileUtil.stat2Paths(fs.listStatus(tfidfVectorPath, PathFilters.logsCRCFilter()));
        } else {
            FileStatus[] inputPaths = fs.globStatus(tfidfVectorPath);
            pathArr = new Path[inputPaths.length];
            int i = 0;
            for (FileStatus fstatus : inputPaths) {
                pathArr[i++] = fstatus.getPath();
            }
        }

        String dictionaryType = getOption("dictionaryType", "text");
        String[] dictionary = null;
        if (hasOption("dictionary")) {
            String dictFile = getOption("dictionary");
            switch (dictionaryType) {
                case "text":
                    dictionary = VectorHelper.loadTermDictionary(new File(dictFile));
                    break;
                case "sequencefile":
                    dictionary = VectorHelper.loadTermDictionary(conf, dictFile);
                    break;
                default:
                    //TODO: support Lucene's FST as a dictionary type
                    throw new IOException("Invalid dictionary type: " + dictionaryType);
            }
        }

        for (Path path : pathArr) {
            Path outputFile = new Path(path.getParent().getParent(), "chi-vectors/" + path.getName());
            if (fs.exists(outputFile)) {
                fs.delete(outputFile, true);
            }
            Writer writer = SequenceFile.createWriter(conf, Writer.keyClass(Text.class),
                    Writer.valueClass(VectorWritable.class), Writer.file(outputFile));

            SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<>(path, true, conf);
            Iterator<Pair<Writable, Writable>> iterator = iterable.iterator();
            while (iterator.hasNext()) {
                Pair<Writable, Writable> record = iterator.next();
                Writable keyWritable = record.getFirst();
                Writable valueWritable = record.getSecond();
                Vector vector = ((VectorWritable) valueWritable).get();
                Vector chiVector = vector.clone();
                for (Element e : vector.nonZeroes()) {
                    double chiVal = 0.0;
                    if (chiMap.containsKey(dictionary[e.index()])) {
                        chiVal = chiMap.get(dictionary[e.index()]);
                    }
                    chiVector.setQuick(e.index(), e.get() * chiVal);
                }
                writer.append(keyWritable, new VectorWritable(chiVector));
            }

            writer.close();
        }

        return 0;
    }

    private Map<String, Double> loadCHI(FileSystem fs, Path chiPath) throws IOException {
        Map<String, Double> map = Maps.newHashMap();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(chiPath)))) {
            String line = null;
            String[] ss = null;
            while ((line = br.readLine()) != null) {
                ss = line.split(" ");
                map.put(ss[0], Double.valueOf(ss[1]));
            }
        }
        return map;
    }
}
