#!/bin/bash
touch measurements.csv
printf  "strategy,nthreads,npoints,runtime\n" > measurements.csv

for l in 1 2 3
do
	for i in 1 2 3 4 6 8 16 32 64
	do
		for j in 10000000 100000000 1000000000
		do
			for k in 0 1 2 3
			do
				printf "%d,%d,%d," $l $i $j >> measurements.csv
				printf "\n%d,%d,%d:" $l $i $j
				/usr/bin/time -o measurements.csv -a --format="%e" ./mcpi $j $i $l
			done
		done
	done
done
