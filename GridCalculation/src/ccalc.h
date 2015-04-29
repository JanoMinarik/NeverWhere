#ifndef __ccalc_H_INCLUDED__
#define __ccalc_H_INCLUDED__
#include "grid.h"

void freeMem(double*, double*, double*, double*);

void calcDensCuda(int, grid);

__global__ void calcDens(int, int, int, double*, double*, double*, double*);
__global__ void calcDensFast(int, int, int, double*, double*, double*, double*);
#endif
