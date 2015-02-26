# set compiler
CC = g++

# compiler flags
CFLAGS = -I./src/

# target
TARGET = main.x

# cpp files
FILES = ./src/*.cpp main.cpp

grid: ./src/grid.cpp ./src/atom.cpp main.cpp
	$(CC) -o $(TARGET) $(FILES) $(CFLAGS)
