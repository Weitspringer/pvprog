# Makefile for Writing Make Files Example

# *****************************************************
# Variables to control Makefile operation

CC = g++
CFLAGS = -Wall -g -fopenmp

# ****************************************************
# Targets needed to bring the executable up to date

mcpi: mcpi.o
	$(CC) $(CFLAGS) -o mcpi mcpi.o

# The main.o target can be written more simply

mcpi.o: mcpi.cpp mcpi.h
	$(CC) $(CFLAGS) -c mcpi.cpp

measurement: measurement.o karpFlattMetric.o
	$(CC) $(CFLAGS) -o measurement measurement.o karpFlattMetric.o

measurement.o: measurement.cpp measurement.h
	$(CC) $(CFLAGS) -c measurement.cpp

karpFlattMetric.o: karpFlattMetric.cpp karpFlattMetric.h
	$(CC) $(CFLAGS) -c karpFlattMetric.cpp