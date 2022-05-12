#!/bin/bash

source "./myct_run_limits.sh"
source "./myct_isolation.sh"

chmod u+x myct.sh
chmod u+x myct_run_limits.sh
chmod u+x test/memory_test.sh
chmod u+x test/cpu_test.sh

./myct.sh init /opt/container/myct
./myct.sh map /opt/container/myct /opt/workspace /opt/workspace

# test: limit memory
controller="memory"
key="limit_in_bytes"
value="10M"
./myct.sh run /opt/container/myct --namespace "test-ns" --limit "$controller.$key=$value" ./test/memory_test.sh

# test: unknown option
./myct.sh run /opt/container/myct -n "test-ns" -m -l "$controller.$key=$value" ./test/memory_test.sh

# test: limit cpu
controller="cpu"
key="cfs_quota_us"
value="200000"
./myct.sh run /opt/container/myct -n "test-ns" -l "$controller.$key=$value" ./test/cpu_test.sh

chmod u-x test/memory_test.sh
chmod u-x test/cpu_test.sh
chmod u-x myct_run_limits.sh
chmod u-x myct.sh
