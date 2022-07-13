# Makefile for Writing Make Files Example

# *****************************************************
# Variables to control Makefile operation

CC = nvcc
CFLAGS = -g
NVCC = -g -o

# ****************************************************
# Targets needed to bring the executable up to date

heatmap: main.o utils.o heatmap.o lifecycle.o
	$(CC) $(NVCC) heatmap main.obj utils.obj heatmap.obj lifecycle.obj

# The main.o target can be written more simply

main.o: main.cu
	$(CC) $(CFLAGS) -c main.cu -dc

utils.o: utils.cu
	$(CC) $(CFLAGS) -c utils.cu -dc

heatmap.o: heatmap.cu
	$(CC) $(CFLAGS) -c heatmap.cu -dc

lifecycle.o: lifecycle.cpp
	$(CC) $(CFLAGS) -c lifecycle.cpp -dc