chmod u+x myct.sh

#./myct.sh init /opt/container/myct
./myct.sh map /opt/container/myct /opt/workspace /opt/workspace
./myct.sh run --namespace "test-ns" --limit "test-limit"

./myct.sh run -n "test-ns" -l "test-limit"

./myct.sh run -n "test-ns" -m -l "test-limit"

chmod u-x myct.sh
