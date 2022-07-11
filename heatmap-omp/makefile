# Makefile for Writing Make Files Example

# *****************************************************
# Variables to control Makefile operation

CC = g++
CFLAGS = -Wall -g -fopenmp

# ****************************************************
# Targets needed to bring the executable up to date

heatmap: main.o utils.o heatmap.o lifecycle.o
	$(CC) $(CFLAGS) -o heatmap main.o utils.o heatmap.o lifecycle.o

# The main.o target can be written more simply

main.o: main.cpp main.h
	$(CC) $(CFLAGS) -c main.cpp main.h

utils.o: utils.cpp utils.h
	$(CC) $(CFLAGS) -c utils.cpp utils.h

heatmap.o: heatmap.cpp heatmap.h
	$(CC) $(CFLAGS) -c heatmap.cpp heatmap.h

lifecycle.o: lifecycle.cpp lifecycle.h
	$(CC) $(CFLAGS) -c lifecycle.cpp lifecycle.h