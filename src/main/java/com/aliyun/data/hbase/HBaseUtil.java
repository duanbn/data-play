package com.aliyun.data.hbase;

import java.math.BigDecimal;

import org.apache.hadoop.hbase.util.Bytes;

public class HBaseUtil {

    public static byte[] toBytes(Object value) {
        if (value instanceof Boolean) {
            return Bytes.toBytes((Boolean) value);
        } else if (value instanceof Short) {
            return Bytes.toBytes((Short) value);
        } else if (value instanceof Integer) {
            return Bytes.toBytes((Integer) value);
        } else if (value instanceof Long) {
            return Bytes.toBytes((Long) value);
        } else if (value instanceof Float) {
            return Bytes.toBytes((Float) value);
        } else if (value instanceof Double) {
            return Bytes.toBytes((Double) value);
        } else if (value instanceof String) {
            return Bytes.toBytes((String) value);
        } else if (value instanceof BigDecimal) {
            return Bytes.toBytes((BigDecimal) value);
        } else {
            throw new IllegalArgumentException("unsupport column value type " + value.getClass());
        }
    }

}
