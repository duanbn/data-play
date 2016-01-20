package com.aliyun.classifier.svm;

import org.junit.Test;

import com.aliyun.app.discovery.shared.utility.web.HtmlUtil;
import com.aliyun.app.discovery.shared.utility.web.WebUtil;

public class LibSVMClassifierTest {

    private LibSVMClassifier classifier = LibSVMClassifier.getInstance();

    @Test
    public void testOne() throws Exception {
        String htmlSource = WebUtil.download("http://www.jd.com/");
        String cs = classifier.classify(HtmlUtil.extract(htmlSource).getSimipleContent());
        System.out.println(cs);
    }

    @Test
    public void testFull() throws Exception {
        String htmlSource = WebUtil.download("http://www.jd.com/");
        String[] cs = classifier.classifyFull(HtmlUtil.extract(htmlSource).getSimipleContent());
        for (String c : cs) {
            System.out.println(c);
        }
    }

}
