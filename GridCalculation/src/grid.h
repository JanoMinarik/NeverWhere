#ifndef __grid_H_INCLUDED__
#define __grid_H_INCLUDED__
#include "atom.h"

class grid{
  public:
    int noPoints;
    atom gridAtom;
    /// functions
    void setGrid(int);
    void unsetGrid();
    void setCoord(double);
    void setCoord(double, double);
    void setCoord(int, double, double, double);
    void initAtom(int, int);
    void setShell(int, int, double, double, double);
    void setAtom(int[] ,
                 double[], double[], double[],
                 double[], double[]);
    void setShellNumber(int[]);
    void setShellFunction(int[]);
    void setExp(double[]);
    void setCoef(double[]);

    void calcGrid();
    void calcDensity();
    void printGrid();
    void printFullGrid();

    double *xCoord;
    double *yCoord;
    double *zCoord;
    double **gridValue;
    double *gridDensity;
    double getR(int, int);
    double getValue(int, double);
    int getNoFnc();
};

#endif
