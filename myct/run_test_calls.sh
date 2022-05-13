#!/bin/bash

source "./myct_run_limits.sh"
source "./myct_isolation.sh"

declare -r CONTAINER_PATH=./container
declare -r MAP_SOURCE=./test
declare -r MAP_TARGET=/opt/workspace

chmod u+x myct.sh
chmod u+x myct_run_limits.sh
chmod u+x test/cpu_test.sh

./myct.sh init $CONTAINER_PATH
./myct.sh map $CONTAINER_PATH $MAP_SOURCE $MAP_TARGET

# test: limit cpu
controller="cpu"
key="cfs_quota_us"
value=$(echo "$(cat /sys/fs/cgroup/cpu/myct/cpu.cfs_period_us) * 0.2" | bc -l)
value=${value%.*}
./myct.sh run $CONTAINER_PATH -l "$controller.$key=$value" $MAP_TARGET/cpu_test.sh

sleep 10

ps u -C cpu_test.sh
./myct.sh unmap $CONTAINER_PATH $MAP_TARGET

chmod u-x test/cpu_test.sh
chmod u-x myct_run_limits.sh
chmod u-x myct.sh

