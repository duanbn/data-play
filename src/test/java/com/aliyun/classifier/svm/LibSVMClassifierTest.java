package com.aliyun.classifier.svm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.junit.Test;

import com.aliyun.app.discovery.shared.utility.web.HtmlUtil;
import com.aliyun.app.discovery.shared.utility.web.WebUtil;

public class LibSVMClassifierTest {

    private LibSVMClassifier classifier = LibSVMClassifier.getInstance();

    @Test
    public void testOne() throws Exception {
        String htmlSource = WebUtil.download("http://www.jd.com/");
        String[] cs = classifier.classify(HtmlUtil.extract(htmlSource).getSimipleContent());
        String cLine = "";
        for (String c : cs) {
            cLine += c + " ";
        }
        System.out.println(cLine);
    }

    @Test
    public void testClassify() throws Exception {
        try (FileReader fr = new FileReader(new File("/Users/shanwei/workspace/classifier/seed.csv"));
                BufferedWriter bw = new BufferedWriter(new FileWriter(new File(
                        "/Users/shanwei/workspace/classifier/seed.out")))) {
            List<String> lines = IOUtils.readLines(fr);
            String[] ss = null;
            for (String line : lines) {
                try {
                    ss = line.split(",");

                    String htmlSource = WebUtil.download("http://" + ss[1]);
                    String[] cs = classifier.classify(HtmlUtil.extract(htmlSource).getSimipleContent());
                    String domain = ss[1];
                    String cLine = "";
                    for (String c : cs) {
                        cLine += c + " ";
                    }
                    bw.write(StringUtils.rightPad(domain, 40) + cLine);
                    bw.newLine();
                    bw.flush();
                } catch (Exception e) {
                    System.out.println(e.getMessage());
                }
            }
        }
    }

}
