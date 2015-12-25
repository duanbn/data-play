#!/bin/sh

basedir=`dirname $0`
basedir=`cd $basedir/..; pwd;`

libdir=$basedir/lib
configdir=$basedir/conf

#load config file
classpath=$classpath:$configdir
#load jar file
classpath=$classpath:$libdir/*


main=com.aliyun.classifier.svm.cmdline.LibSVMMain

opts="-Xms128m"
sysargs="-Dsystem.prop.basedir=$basedir"
exec java $opts -cp $classpath $sysargs $main -i $1 -o $2
