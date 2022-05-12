#!/bin/bash

file=$(cat data.csv)

for line in $file
do
    echo -e "$line\n"
done
