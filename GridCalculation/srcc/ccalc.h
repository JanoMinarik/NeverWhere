#ifndef __ccalc_H_INCLUDED__
#define __ccalc_H_INCLUDED__
#include "grid.h"

void cpyInitial(double*, double*, double*, double*,
                           double*, double*, double*,
                           int, int);
void freeMem(double*, double*, double*, double*);
void cpyResult(double*, double*, int);

void calcDensCuda(int, int, grid);

__global__ void calcDens(int, int, int, double*, double*, double*, double*);
__global__ void calcDensFast(int, int, int, double*, double*, double*, double*);
#endif
