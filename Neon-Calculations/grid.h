#ifndef __grid_H_INCLUDED__
#define __grid_H_INCLUDED__
#include "neon.h"

class Grid
{
public:
    int numNeons;
    int numPoints;
    int numFnc;
    Neon *gridNeons;
    void setGrid(int, int);
    void unsetGrid();
    void setCoord(int);
    void setCoord(int, double, double, double);
    void setNeon(int, double, double, double);
    void calcGrid();
    void printGrid();

protected:
    double *xCoord;
    double *yCoord;
    double *zCoord;
    double **gridValue;
    int getNumFnc();
    double getR(int, int, int);
    double getValue(int, int, double);
};

#endif
