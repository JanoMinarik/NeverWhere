#ifndef __grid_H_INCLUDED__
#define __grid_H_INCLUDED__
#include "neon.h"

class Grid : Neon
{
public:
    int numPoints;
    int numFnc;
    void setGrid(int);
    void unsetGrid();
    void setCoord();
    void setCoord(int, double, double, double);
    void calcGrid();
    void printGrid();
    double getValue(int curFnc, double R);

protected:
    double *xCoord;
    double *yCoord;
    double *zCoord;
    double **gridValue;
    int getNumFnc();
    double getR(int, int);
    double getValue(int, int);
};

#endif
