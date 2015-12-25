package com.aliyun.data.hbase;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Admin;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.Delete;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Table;

import com.google.common.collect.Lists;

public class HBaseDML implements HBaseAccess {

    private Connection conn;

    HBaseDML(Connection conn) {
        this.conn = conn;
    }

    public void close() throws IOException {
        this.conn.close();
    }

    public void delete(String tableName, List<? extends Object> rowKeys) {
        TableName tn = TableName.valueOf(tableName);
        try (Admin admin = conn.getAdmin(); Table table = conn.getTable(tn)) {
            if (admin.tableExists(tn)) {
                List<Delete> deletes = Lists.newArrayList();
                Delete delete = null;
                for (Object rowKey : rowKeys) {
                    delete = new Delete(HBaseUtil.toBytes(rowKey));
                    deletes.add(delete);
                }
                table.delete(deletes);
            } else {
                throw new HBaseOperationException(tableName + " not exists");
            }
        } catch (IOException e) {
            throw new HBaseOperationException(e);
        }
    }

    public void insertOrUpdate(String tableName, List<HBaseRecord> records) {
        TableName tn = TableName.valueOf(tableName);
        try (Admin admin = conn.getAdmin(); Table table = conn.getTable(tn)) {
            if (admin.tableExists(tn)) {
                List<Put> puts = Lists.newArrayListWithCapacity(records.size());
                Put put = null;
                for (HBaseRecord record : records) {
                    put = new Put(record.getRowKey());
                    for (HBaseRecord.Cell cell : record.getCells()) {
                        put.addColumn(cell.getColumnFamily(), cell.getQualifier(), cell.getValue());
                    }
                    puts.add(put);
                }
                table.put(puts);
            } else {
                throw new HBaseOperationException(tableName + " not exists");
            }
        } catch (IOException e) {
            throw new HBaseOperationException(e);
        }
    }

}
