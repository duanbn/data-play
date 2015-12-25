package com.aliyun.data.hbase;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.hbase.util.Bytes;

import com.google.common.collect.Lists;

public class HBaseRecord {

    private byte[]     rowKey;

    private List<Cell> cells = Lists.newArrayList();

    private HBaseRecord(Object rowKey) {
        this.rowKey = HBaseUtil.toBytes(rowKey);
    }

    public static HBaseRecord valueOf(Object rowKey) {
        return new HBaseRecord(rowKey);
    }

    public static HBaseRecord valueOf(Object rowKey, List<Cell> cells) {
        HBaseRecord record = valueOf(rowKey);
        record.append(cells);
        return record;
    }

    public void append(Cell cell) {
        this.cells.add(cell);
    }

    public void append(List<Cell> cells) {
        this.cells.addAll(cells);
    }

    public void clear() {
        this.cells.clear();
    }

    public static class Cell {
        private byte[] columnFamily;
        private byte[] qualifier;
        private byte[] value;

        private Cell(byte[] columnFamily, byte[] qualifier, byte[] value) {
            this.columnFamily = columnFamily;
            this.qualifier = qualifier;
            this.value = value;
        }

        public static Cell valueOf(String columnFamily, String qualifier, Object value) throws IOException {
            return new Cell(Bytes.toBytes(columnFamily), Bytes.toBytes(qualifier), HBaseUtil.toBytes(value));
        }

        public byte[] getColumnFamily() {
            return this.columnFamily;
        }

        public byte[] getQualifier() {
            return qualifier;
        }

        public byte[] getValue() {
            return value;
        }
    }

    public byte[] getRowKey() {
        return rowKey;
    }

    public void setRowKey(byte[] rowKey) {
        this.rowKey = rowKey;
    }

    public List<Cell> getCells() {
        return cells;
    }

}
