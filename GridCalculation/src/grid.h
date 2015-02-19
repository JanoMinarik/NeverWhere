#ifndef __grid_H_INCLUDED__
#define __grid_H_INCLUDED__
#include "atom.h"

class grid{
  public:
    // grid properites
    int noPoints;
    int noAOs;
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
    void setDensityMatrix(double[15][15]);
    void setShellNumber(int[]);
    void setShellFunction(int[]);
    void setExp(double[]);
    void setCoef(double[]);
    // grid calculation
    void calcGrid();
    void printGrid();
    void printFullGrid();
    void printDensityMatrix();
    void printGridInfo();
    // grid variables
    double *xCoord;
    double *yCoord;
    double *zCoord;
    double **gridValue;
    double *gridDensity;
    double **densityMatrix;
    // support functions
    double getR(int, int);
    double getValue(int, double);
    int getNoFnc();
    void calcDensity();
};

#endif
