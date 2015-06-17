# set compiler
#CXX = g++
CXX = icpc

# compiler flags
PFLAGS = -mkl
#PFLAGS = 
CFLAGS = -I./src/

# target
TARGET = main.x

# cpp files
FILES = ./src/*.cpp main.cpp

# run compilation
grid: ./src/grid.cpp ./src/atom.cpp main.cpp

	$(CXX) $(PFLAGS) -o $(TARGET) $(FILES) $(CFLAGS)
