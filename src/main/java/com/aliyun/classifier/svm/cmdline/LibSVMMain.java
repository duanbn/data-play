package com.aliyun.classifier.svm.cmdline;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;

import com.aliyun.app.discovery.shared.utility.web.HtmlUtil;
import com.aliyun.app.discovery.shared.utility.web.WebUtil;
import com.aliyun.classifier.Config;
import com.aliyun.classifier.svm.LibSVMClassifier;
import com.google.common.collect.Queues;

public class LibSVMMain extends Config {

    private static final BlockingQueue<String> outQ = Queues.newLinkedBlockingQueue();

    public static void main(String[] args) throws Exception {

        CommandLineParser parser = new PosixParser();
        Options options = new Options();
        options.addOption("i", "input", true, "输入文件, 格式：一行一个url");
        options.addOption("o", "output", true, "输出文件");

        HelpFormatter formatter = new HelpFormatter();
        String helpInfo = "svm classifier, dependency on JDK1.7+";

        CommandLine commandLine = null;
        try {
            commandLine = parser.parse(options, args);
        } catch (ParseException e) {
            formatter.printHelp(helpInfo, options);
            System.exit(-1);
        }

        File input = null, output = null;
        if (commandLine.hasOption("input")) {
            input = new File(commandLine.getOptionValue("input"));
        } else {
            formatter.printHelp(helpInfo, options);
            System.exit(-1);
        }
        if (commandLine.hasOption("output")) {
            output = new File(commandLine.getOptionValue("output"));
        } else {
            formatter.printHelp(helpInfo, options);
            System.exit(-1);
        }

        List<String> lines = null;
        try (BufferedReader br = new BufferedReader(new FileReader(input))) {
            lines = IOUtils.readLines(br);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        final LibSVMClassifier classifier = LibSVMClassifier.getInstance();

        final CountDownLatch cdl = new CountDownLatch(lines.size());
        for (final String domain : lines) {
            threadPool.submit(new Runnable() {
                @Override
                public void run() {
                    try {
                        String url = null;
                        if (domain.startsWith("http://")) {
                            url = domain;
                        } else {
                            url = "http://" + domain;
                        }
                        String htmlSource = WebUtil.download(url);
                        long start = System.currentTimeMillis();
                        String oneTag = classifier.classify(domain, HtmlUtil.extract(htmlSource).getSimipleContent());
                        //                        String[] tags = classifier.classifyFull(HtmlUtil.extract(htmlSource).getSimipleContent());
                        //                        String tagLine = "[" + oneTag + "] ";
                        //                        for (String tag : tags) {
                        //                            tagLine += tag + " ";
                        //                        }
                        outQ.offer(StringUtils.rightPad(url, 50) + StringUtils.rightPad(oneTag, 30)
                                + (System.currentTimeMillis() - start) + "ms");
                    } catch (Exception e) {
                        System.out.println(e.getMessage());
                    } finally {
                        cdl.countDown();
                    }
                }
            });
        }

        final BufferedWriter bw = new BufferedWriter(new FileWriter(output));
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        String line = outQ.take();
                        bw.write(line);
                        bw.newLine();
                        bw.flush();
                    } catch (Exception e) {
                        System.out.println(e.getMessage());
                    }
                }
            }
        }).start();

        cdl.await();
        bw.close();

        System.exit(0);

    }
}
