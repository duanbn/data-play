package com.aliyun.classifier.svm;

import org.junit.Test;

import com.aliyun.app.discovery.shared.utility.web.HtmlUtil;
import com.aliyun.app.discovery.shared.utility.web.WebUtil;

public class LibSVMClassifierTest {

    private LibSVMClassifier classifier = LibSVMClassifier.getInstance();

    @Test
    public void testOne() throws Exception {
        String domain = "http://www.jd.com/";
        String htmlSource = WebUtil.download(domain);
        String cs = classifier.classify(domain, HtmlUtil.extract(htmlSource).getSimipleContent());
        System.out.println(cs);
    }

    @Test
    public void testFull() throws Exception {
        //        String domain = "http://www.jd.com/";
        //        String htmlSource = WebUtil.download(domain);
        //        String[] cs = classifier.classifyFull(domain, HtmlUtil.extract(htmlSource).getSimipleContent());
        //        for (String c : cs) {
        //            System.out.println(c);
        //        }
    }

}
