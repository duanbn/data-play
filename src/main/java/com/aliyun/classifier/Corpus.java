package com.aliyun.classifier;

import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;

public class Corpus {

    public int                           N                 = 0;

    public Multimap<String, Set<String>> categoryTokenized = ArrayListMultimap.create();

    public Multiset<String>              featureDict       = ConcurrentHashMultiset.create();

    public List<Feature>                    features          = Lists.newArrayList();

    public Map<String, Integer>          df                = Maps.newHashMap();

    public Map<String, Integer>          categoryDocCount  = Maps.newHashMap();

}
