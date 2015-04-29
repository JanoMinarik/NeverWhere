# set compiler
#CC = g++
CC = icc

# compiler flags
PFLAGS = -mkl
CFLAGS = -I./src/

# target
TARGET = main.x

# cpp files
FILES = ./src/*.cpp main.cpp

# run compilation
grid: ./src/grid.cpp ./src/atom.cpp main.cpp

	$(CC) $(PFLAGS) -o $(TARGET) $(FILES) $(CFLAGS)
