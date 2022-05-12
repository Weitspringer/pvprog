chmod u+x myct.sh

#./myct.sh init /opt/container/myct
./myct.sh map /opt/container/myct /opt/workspace /opt/workspace
./myct.sh run --namespace "test-ns" --limit "test-limit"

./myct.sh run -n "test-ns" -l "test-limit"

./myct.sh run -n "test-ns" -m -l "test-limit"


chmod u-x myct.sh


chmod u+x test/limit_test.sh
chmod u+x myct_run_limits.sh


controller="memory"
key="limit_in_bytes"
value="10M"

echo "limit container resources"
# limit container resources
myct_run_limits::limit $controller $key $value

echo "apply limits on container"
# apply limits on container
myct_run_limits::add_process $controller $$
./test/memory_test.sh


controller="cpu"
key="cfs_quota_us"
value="200000"
echo "limit container resources"
myct_run_limits::limit $controller $key $value
myct_run_limits::limit $controller cfs_period_us 1000000
myct_run_limits::add_process $controller $$
./test/cpu_test.sh

chmod u-x test/memory_test.sh
chmod u-x test/cpu_test.shh
chmod u-x myct_run_limits.sh

