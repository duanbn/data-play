package com.aliyun.data.hbase;

import java.io.IOException;

import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Admin;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.util.Bytes;

public class HBaseDDL implements HBaseAccess {

    private Connection conn;

    HBaseDDL(Connection conn) {
        this.conn = conn;
    }

    @Override
    public void close() throws IOException {
        this.conn.close();
    }

    public HTableDescriptor[] listTables() throws Exception {
        try (Admin admin = conn.getAdmin()) {
            return admin.listTables();
        }
    }

    public void deleteTable(String... tableNames) throws Exception {
        try (Admin admin = conn.getAdmin();) {
            for (String tableName : tableNames) {
                TableName tn = TableName.valueOf(tableName);
                if (admin.tableExists(tn)) {
                    admin.disableTable(tn);
                    admin.deleteTable(tn);
                }
            }
        }
    }

    public void addColumnFamily(String tableName, String... columnFamilys) throws Exception {
        TableName tn = TableName.valueOf(tableName);
        try (Admin admin = conn.getAdmin();) {
            if (admin.tableExists(tn)) {
                HTableDescriptor hTableDesc = admin.getTableDescriptor(tn);
                for (String columnFamily : columnFamilys) {
                    hTableDesc.addFamily(new HColumnDescriptor(columnFamily));
                }
                admin.disableTable(tn);
                admin.modifyTable(tn, hTableDesc);
                admin.enableTable(tn);
            } else {
                throw new IllegalArgumentException(tableName + " not exists");
            }
        }
    }

    public void deleteColumnFamily(String tableName, String... columnFamilys) throws Exception {
        TableName tn = TableName.valueOf(tableName);
        try (Admin admin = conn.getAdmin();) {
            if (admin.tableExists(tn)) {
                HTableDescriptor hTableDesc = admin.getTableDescriptor(tn);
                for (String columnFamily : columnFamilys) {
                    hTableDesc.removeFamily(Bytes.toBytes(columnFamily));
                }
                admin.disableTable(tn);
                admin.modifyTable(tn, hTableDesc);
                admin.enableTable(tn);
            } else {
                throw new IllegalArgumentException(tableName + " not exists");
            }
        }
    }

    public void createTable(String tableName, String... columnFamilys) throws Exception {
        TableName tn = TableName.valueOf(tableName);
        try (Admin admin = conn.getAdmin();) {
            if (!admin.tableExists(tn)) {
                HTableDescriptor hTableDesc = new HTableDescriptor(tn);
                for (String columnFamily : columnFamilys) {
                    hTableDesc.addFamily(new HColumnDescriptor(columnFamily));
                }
                admin.createTable(hTableDesc);
            } else {
                throw new IllegalArgumentException(tableName + " has exists");
            }
        }
    }
}
