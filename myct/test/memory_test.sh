#!/bin/bash

echo "let'S try  ./test/data_limit_ok.csv"
file=$(cat ./test/data_limit_ok.csv)
echo "success reading ./test/data_limit_ok.csv"

echo "let'S try  ./test/data_limit_exceeded.csv"
file=$(cat ./test/data_limit_exceeded.csv)
echo "success reading ./test/data_limit_exceeded.csv -> should not have happened"
