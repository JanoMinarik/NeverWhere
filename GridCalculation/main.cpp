#include "atom.h"
#include "grid.h"
#define CUDA_ENABLED 0
#define MKL_ENABLED 0

#if CUDA_ENABLED
  #include "ccalc.h"
#endif
//extern static double precision = 1e-12
  
int main(){
  grid myGrid;
  int shellNos [] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, };
  int shellFncs [] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2, 3, 4, 1, 1, };
  double shellCoefs [] = {0.81328399, 1.50832394, 2.53133644, 3.69806595, 4.28062152, 3.06933306, 0.859136106, 0.0156504324, -0.000872387333, -0.189545807, -0.360541657, -0.59043758, -0.942505698, -1.12138075, -1.19964549, -0.356398904, 0.589163852, 0.234803397, 0.415422412, 4.30502365, 3.39674885, 1.40249651, 0.227278748, 0.498789898, 11.347505, };
  double shellExps [] = {17880.0, 2683.0, 611.5, 173.5, 56.64, 20.42, 7.81, 1.653, 0.4869, 17880.0, 2683.0, 611.5, 173.5, 56.64, 20.42, 7.81, 1.653, 0.4869, 0.4869, 28.39, 6.27, 1.695, 0.4317, 0.4317, 2.202, };

  double dMat[15][15] = {
{1.004984, 0.005112, -0.013033, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.005112, 0.933506, 0.034989, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{-0.013033, 0.034989, 0.001485, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 1.000144, 0.000000, 0.000000, -0.000091, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 0.000000, 1.000144, 0.000000, 0.000000, -0.000091, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000144, 0.000000, 0.000000, -0.000091, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, -0.000091, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 0.000000, -0.000091, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.000091, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
{0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, },
  };

  double *pDMat = &dMat[0][0];
  myGrid.initAtom(6, 25);
  myGrid.setGrid(16496);
  myGrid.setShellNumber(shellNos);
  myGrid.setShellFunction(shellFncs);
  myGrid.setExp(shellExps);
  myGrid.setCoef(shellCoefs);
  myGrid.setDensityMatrix(pDMat);
  myGrid.setShell(0, 0, 0.0, 0.0, 0.0);
  myGrid.setShell(1, 0, 0.0, 0.0, 0.0);
  myGrid.setShell(2, 0, 0.0, 0.0, 0.0);
  myGrid.setShell(3, 1, 0.0, 0.0, 0.0);
  myGrid.setShell(4, 1, 0.0, 0.0, 0.0);
  myGrid.setShell(5, 2, 0.0, 0.0, 0.0);

  myGrid.setCoordFile((char*)"./input/neon-dz/grid.txt");

  myGrid.calcGrid(1, 100); // (x, y): x = computation method, y = number of points in a batch
                           // methods: 1 - sequential, 2 - vector wise (use MKL if able), 3 - batch (use MKL if able)
                           // 4 - no calculation of density, used for cuda
  myGrid.printGridInfo();  
 
#if MKL_ENABLED
    myGrid.calcGrid(2, 100);
    myGrid.printGridInfo();
    myGrid.calcGrid(3, 100);
    myGrid.printGridInfo();
#endif

#if CUDA_ENABLED
    myGrid.calcGrid(4, 100);
    calcDensCuda(100, &myGrid); // (x, grid): x - no. of GPU cores enabled, grid - name of grid 
    myGrid.printGridInfo();
#endif 
 
  //myGrid.printGrid();
  myGrid.printFullGrid();
  return 0;
}