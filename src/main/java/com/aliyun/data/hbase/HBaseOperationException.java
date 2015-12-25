package com.aliyun.data.hbase;

public class HBaseOperationException extends RuntimeException {

    /**
     * 
     */
    private static final long serialVersionUID = -8768414968666879130L;

    public HBaseOperationException(Throwable t) {
        super(t);
    }

    public HBaseOperationException(String msg) {
        super(msg);
    }

}
