# Makefile for Writing Make Files Example

# *****************************************************
# Variables to control Makefile operation

CC = g++
CFLAGS = -Wall -g

# ****************************************************
# Targets needed to bring the executable up to date

heatmap: heatmap.o
	$(CC) $(CFLAGS) -o heatmap heatmap.o

# The main.o target can be written more simply

heatmap.o: heatmap.cpp
	$(CC) $(CFLAGS) -c heatmap.cpp


test_run:
	./heatmap 3 3 2 headmap_random_3.csv