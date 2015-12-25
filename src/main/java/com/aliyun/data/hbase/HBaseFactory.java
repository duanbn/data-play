package com.aliyun.data.hbase;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HConstants;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.ConnectionFactory;

public class HBaseFactory {

    private static volatile HBaseFactory  instance;
    private static volatile Configuration conf;

    static {
        conf = HBaseConfiguration.create();
        conf.set(HConstants.HBASE_CLIENT_SCANNER_TIMEOUT_PERIOD, "120000");
        instance = new HBaseFactory();
    }

    public static HBaseFactory getInstance() {
        return instance;
    }

    public void release(HBaseAccess hbaseAccess) {
        try {
            hbaseAccess.close();
        } catch (IOException e) {
            throw new HBaseOperationException(e);
        }
    }

    public HBaseDML createDML() {
        Connection conn;
        try {
            conn = ConnectionFactory.createConnection(conf);
            return new HBaseDML(conn);
        } catch (IOException e) {
            throw new HBaseOperationException(e);
        }
    }

    public HBaseDDL createDDL() {
        Connection conn;
        try {
            conn = ConnectionFactory.createConnection(conf);
            return new HBaseDDL(conn);
        } catch (IOException e) {
            throw new HBaseOperationException(e);
        }
    }

}
