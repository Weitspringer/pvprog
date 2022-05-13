#!/bin/bash

source "./myct_run_limits.sh"
source "./myct_isolation.sh"

declare -r CONTAINER_PATH=./container/myct
declare -r MAP_SOURCE=./test
declare -r MAP_TARGET=/opt/workspace

chmod u+x myct.sh
chmod u+x myct_run_limits.sh
chmod u+x test/memory_test.sh
chmod u+x test/cpu_test.sh

./myct.sh init $CONTAINER_PATH
./myct.sh map $CONTAINER_PATH $MAP_SOURCE $MAP_TARGET

# test: limit memory
controller="memory"
key="limit_in_bytes"
value="10M"
./myct.sh run $CONTAINER_PATH --namespace "test-ns" --limit "$controller.$key=$value" $MAP_TARGET/memory_test.sh

# test: unknown option
./myct.sh run $CONTAINER_PATH -n "test-ns" -m -l "$controller.$key=$value" $MAP_TARGET/memory_test.sh

# test: limit cpu
controller="cpu"
key="cfs_quota_us"
value="200000"
./myct.sh run $CONTAINER_PATH -n "test-ns" -l "$controller.$key=$value" $MAP_TARGET/cpu_test.sh


./myct.sh unmap $CONTAINER_PATH $MAP_TARGET

chmod u-x test/memory_test.sh
chmod u-x test/cpu_test.sh
chmod u-x myct_run_limits.sh
chmod u-x myct.sh
