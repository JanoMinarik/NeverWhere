#ifndef __grid_H_INCLUDED__
#define __grid_H_INCLUDED__
#include "atom.h"

class grid{
  public:
    // grid properites
    int noPoints;
    int noAOs;
    double atomDensity;
    double clcTm;
    atom gridAtom;
    /// functions
    void setGrid(int);
    void unsetGrid();
    void setCoord(double);
    void setCoord(double, double);
    void setCoord(int, double, double, double, double);
    void setCoordFile(char[]);
    void initAtom(int, int);
    void setShell(int, int, double, double, double);
    void setAtom(int[] ,
                 double[], double[], double[],
                 double[], double[]);
    void setDensityMatrix(double*);
    void setShellNumber(int[]);
    void setShellFunction(int[]);
    void setExp(double[]);
    void setCoef(double[]);
    // grid calculation
    void calcGrid(int, int);
    void printGrid();
    void printFullGrid();
    void printDensityMatrix();
    void printGridInfo();
    void printGridAtom();
    void printCorner(double*, int, int); 
   // grid variables
    double *xCoord;
    double *yCoord;
    double *zCoord;
    double *weight;
    double **gridValue;
    double *gridDensity;
    double **densityMatrix;
    // support functions
    double getR(int, int);
    double getR2(int);
    double getValue(int, double);
    int getNoFnc();
    void calcDensity();
    void calcDensityScr();
    void calcDensityBatch(int);
    void densityScreening(double*, double*, int);
    void cleanArr(double*, int, int);
    double tempArrMull(double*, double*, int);
    void vectorwiseProduct(double*, double*, int, int);
};

#endif
