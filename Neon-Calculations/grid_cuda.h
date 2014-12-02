#ifndef __grid_cuda_H_INCLUDED__
#define __grid_cuda_H_INCLUDED__
#include "Neon.h"

struct gridCuda {
    int numNeons;
    int numPoints;
    int numFnc;
    Neon *gridNeons;
    /// values
    double *xCoord;
    double *yCoord;
    double *zCoord;
    double **gridValue;
};

struct devGrid {
    Neon *gridNeons;
    double *xCoord;
    double *yCoord;
    double *zCoord;
    double **gridValue;
};

/// functions for grid manipulations
void initGrid( gridCuda, int, int );
void calculateGrid( gridCuda );
void printGrid( );

/// kernel functions

/// supporting functions
void host2device( );
void device2host( );
void initDevices( );
void freeDevices( );
int getNumFnc( Neon );

#endif
