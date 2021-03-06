#include "grid.h"
#include "atom.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
#define MKL_ENABLED 0
#if MKL_ENABLED
  #include "mkl.h"
#endif

#define PI 3.1415926
#define min(x,y) ((x < y) ? x : y)

/// grid memory alocation/ free.
void grid::setGrid(int nPn)
{
    noPoints = nPn;
    xCoord = new double[noPoints];
    yCoord = new double[noPoints];
    zCoord = new double[noPoints];
    weight = new double[noPoints];
    gridDensity = new double[noPoints];
    gridValue = new double*[noPoints];
    
    for( int i = 0; i < noPoints; i++ )
    {
        gridValue[i] = new double[noAOs];
    }
    densityMatrix = new double*[noAOs];
    for( int i = 0; i < noAOs; i++ ) {
        densityMatrix[i] = new double[noAOs];
    }
}

void grid::unsetGrid()
{
    delete [] xCoord;
    delete [] yCoord;
    delete [] zCoord;
    delete [] gridDensity;

    for( int i = 0; i < noPoints; i++ )
    {
        delete [] gridValue[i];
    }
    delete gridValue;

    for( int i = 0; i < getNoFnc(); i++ ){
        delete [] densityMatrix[i];
    }
    delete densityMatrix;
}

/// set grid
void grid::initAtom(int noShl, int noFnc){
  gridAtom.noShl = noShl;
  gridAtom.noFnc = noFnc;
  gridAtom.atomShell = new shell[noShl];
  gridAtom.shellNumber = new int[noFnc];
  gridAtom.shellFunction = new int[noFnc];
  gridAtom.coef = new double[noFnc];
  gridAtom.exp = new double[noFnc];
  noAOs = getNoFnc();
}

void grid::setShell(int curShell, int ang, double x, double y, double z){
  gridAtom.atomShell[curShell].ang = ang;
  gridAtom.atomShell[curShell].x = x;
  gridAtom.atomShell[curShell].y = y;
  gridAtom.atomShell[curShell].z = z;
}

void grid::setAtom(int angs[],
                   double xs[], double ys[], double zs[],
                   double exps[], double coefs[]){

  /// dorobit IF atom == NULL RETURN warning

  for(int i=0; i<gridAtom.noShl; i++){
    gridAtom.atomShell[i].ang = angs[i];
    gridAtom.atomShell[i].x = xs[i];
    gridAtom.atomShell[i].y = ys[i];
    gridAtom.atomShell[i].z = zs[i];
  }

  for(int i=0; i<gridAtom.noFnc; i++){
    gridAtom.exp[i] = exps[i];
    gridAtom.coef[i] = coefs[i];
  }
}

void grid::setShellNumber(int arr[]){
  for(int i=0; i<gridAtom.noFnc; i++){
    gridAtom.shellNumber[i] = arr[i];
  }
}

void grid::setShellFunction(int arr[]){
  for(int i=0; i<gridAtom.noFnc; i++){
    gridAtom.shellFunction[i] = arr[i];
  }
}

void grid::setExp(double arr[]){
  for(int i=0; i<gridAtom.noFnc; i++){
    gridAtom.exp[i] = arr[i];
  }
}

void grid::setCoef(double arr[]){
  for(int i=0; i<gridAtom.noFnc; i++){
    gridAtom.coef[i] = arr[i];
  }
}

void grid::setDensityMatrix(double *arr){
  for(int i=0; i<noAOs; i++){
    for(int j=0; j<noAOs; j++){
      densityMatrix[i][j] = *(arr + i*noAOs + j);
    }
  }
}

void grid::setCoord(double range){
    double dR = range/noPoints;
    for( int i = 0; i < noPoints; i++ )
    {
        xCoord[i] = range*(double)rand() / RAND_MAX;
        yCoord[i] = range*(double)rand() / RAND_MAX;
        zCoord[i] = range*(double)rand() / RAND_MAX;
        weight[i] = 4.0*PI*getR2(i)*dR;
    }
}

void grid::setCoord(double start, double end){
    if(start > end)
    {
        double temp = end;
        end = start;
        start  = temp;
    }
    double step = ( end - start )/( noPoints - 1 );

    double dR = (end - start)/noPoints;
    for( int i=0; i<noPoints; i++ )
    {
        xCoord[i] = start + i*step;
        yCoord[i] = start + i*step;
        zCoord[i] = start + i*step;
        weight[i] = 4.0*PI*getR2(i)*dR;
    }
}

void grid::setCoord(int curPt, double x, double y, double z, double w)
{
    xCoord[curPt] = x;
    yCoord[curPt] = y;
    zCoord[curPt] = z;
    weight[curPt] = w;
}

void grid::setCoordFile(char name[]){
  std::ifstream infile (name);
  if(infile.is_open()){
    double x, y, z, w;
    for(int i=0; i<noPoints; i++){
      infile >> x;
      if(x == EOF){break;}
      infile >> y;
      infile >> z;
      infile >> w;
      setCoord(i, x, y, z, w);
    }
  }else{
    std::cout << "Could not open file " << name << "\n";
  }
  infile.close(); 
}

/// print grid
void grid::printGrid(){
  for(int i=0; i<noPoints; i++){
    std::cout << xCoord[i] << " " << yCoord[i] << " " << zCoord[i] << " ";
    std::cout << gridDensity[i] << "\n";
  }
}

void grid::printDensityMatrix(){
  for(int i=0; i<noAOs; i++){
    for(int j=0; j<noAOs; j++){
      std::cout << densityMatrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}

void grid::printGridInfo(){
#if MKL_ENABLED
    std::cout << " Using Intel(R) blas API.\n";
#endif
  std::cout << " == SUCCESS ==\n";
  printf(" calculation time: %.5lf ms\n", clcTm);
  std::cout << " no Points: " << noPoints << "\n";
  std::cout << " no AOs: " << noAOs << "\n";
  printf(" no electrons: %lf\n", atomDensity);
}

void grid::printFullGrid(){
  for(int i=0; i<noPoints; i++){
    std::cout << xCoord[i] << " " << yCoord[i] << " " << zCoord[i] << " ";
    for(int j=0; j<noAOs; j++){
      std::cout << gridValue[i][j] << " ";
    }
    std::cout << gridDensity[i] << "\n";
  }
}

void grid::printGridAtom(){
  std::cout << "shells [no, ang, x, y, z]\n";
  for(int i=0; i<gridAtom.noShl; i++){
    std::cout << i+1 << " ";
    std::cout << gridAtom.atomShell[i].ang << " ";
    std::cout << gridAtom.atomShell[i].x << " ";
    std::cout << gridAtom.atomShell[i].y << " ";
    std::cout << gridAtom.atomShell[i].z << "\n";
  }
  
  std::cout << "contraction functions [shell no, fnc no, exp, coef]\n";
  for(int i=0; i<gridAtom.noFnc; i++){
    std::cout << gridAtom.shellNumber[i] << " " << gridAtom.shellFunction[i] << " ";
    std::cout << gridAtom.exp[i] << " " << gridAtom.coef[i] << "\n";
  }
  std::cout << "====================\n";
}

void grid::printCorner(double *A, int m, int n){
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++)std::cout << A[i*n+j]  << " ";
    std::cout << "\n";
  }
}
/// calculate grid density
void grid::calcGrid(int opt, int pts)
{
    int curValue, curFnc;
    static int LOOP_CNT = 1;
    double value, R;
#if MKL_ENABLED
    double s_init;
#else
    clock_t s_init, s_end;
#endif
    for( int curPt = 0; curPt < noPoints; curPt++ )
    {
        curValue = curFnc = 0;
        for( int j = 0; j < gridAtom.noShl; j++ ) /// j =  curShell
        {
            R = getR(curPt, j);
            value = 0;
            while( gridAtom.shellNumber[curFnc] == (j+1) )
            {
                value += getValue(curFnc, R);
                curFnc++;
            }
            if( gridAtom.atomShell[j].ang == 0 )
            {
                gridValue[curPt][curValue++] = value;
            }

            if( gridAtom.atomShell[j].ang == 1 )
            {
                gridValue[curPt][curValue++] = xCoord[curPt]*value;
                gridValue[curPt][curValue++] = yCoord[curPt]*value;
                gridValue[curPt][curValue++] = zCoord[curPt]*value;
            }

            if( gridAtom.atomShell[j].ang == 2 )
            {
                gridValue[curPt][curValue++] = xCoord[curPt]*xCoord[curPt]*value;
                gridValue[curPt][curValue++] = yCoord[curPt]*yCoord[curPt]*value;
                gridValue[curPt][curValue++] = zCoord[curPt]*zCoord[curPt]*value;
                gridValue[curPt][curValue++] = xCoord[curPt]*yCoord[curPt]*value;
                gridValue[curPt][curValue++] = xCoord[curPt]*zCoord[curPt]*value;
                gridValue[curPt][curValue++] = yCoord[curPt]*zCoord[curPt]*value;
            }
        }
    }
#if MKL_ENABLED 
      s_init = dsecnd();
#else
      s_init = clock(); 
#endif 
    switch(opt){
      case 1:
        calcDensity();
        break;
      case 2:
        calcDensityScr();
        break;
      case 3:
        calcDensityBatch(pts);
        break;
      default:
        break;
    }
#if MKL_ENABLED
      clcTm = (dsecnd() - s_init)/LOOP_CNT*1000;
#else
      s_end = clock();
      clcTm = (double)(s_end-s_init)/CLOCKS_PER_SEC*1000/LOOP_CNT;
#endif
}

void grid::calcDensity(){
  atomDensity = 0.0;
  for(int p=0; p<noPoints; p++){
    gridDensity[p] = 0.0;
    for(int k=0; k<noAOs; k++){
      for(int l=0; l<k; l++){
        gridDensity[p] += 2*densityMatrix[k][l]*gridValue[p][k]*gridValue[p][l];
      }
      gridDensity[p] += densityMatrix[k][k]*gridValue[p][k]*gridValue[p][k];
    }
    gridDensity[p] *= weight[p];
    atomDensity += gridDensity[p];
  }
}

void grid::calcDensityBatch(int pts){
  int start, end, step;
  start = 0; end = min(pts, noPoints);
  double *chi, *X, *DM;
 
#if MKL_ENABLED 
    DM = (double*)mkl_malloc( noAOs*noAOs*sizeof(double), 64 ); 
    chi = (double*)mkl_malloc( pts*noAOs*sizeof(double), 64 );
    X = (double*)mkl_malloc( pts*noAOs*sizeof(double), 64 );
#else
    DM = (double*)malloc( noAOs*noAOs*sizeof(double) );
    chi = (double*)malloc( pts*noAOs*sizeof(double) );
    X = (double*)malloc( pts*noAOs*sizeof(double) );
#endif
 
  DM = &densityMatrix[0][0]; 


  if( chi == NULL || X == NULL || DM == NULL ){
    std::cout << " == ERROR: Memory allocation failed. ==\n\n";
#if MKL_ENABLED
      mkl_free(chi);
      mkl_free(DM);
      mkl_free(X);
#else
      free(chi);
      free(DM);
      free(X);
#endif
    return;
  }

  atomDensity = 0.0;
  for(; end<noPoints;){
    step = end - start;
    for(int i=0; i<step; i++){
      for(int j=0; j<noAOs; j++){
        chi[i*noAOs + j] = gridValue[start+i][j];
      }
    }    

#if MKL_ENABLED
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, step, noAOs, noAOs, 1, chi, noAOs, DM, noAOs, 0, X, noAOs);
#else
      X =  tempMatMull(chi, DM, step, noAOs, noAOs); 
#endif
    vectorwiseProduct(X, chi, start, end);
    
    start = end;
    end = min(end+pts, noPoints);
  }
#if MKL_ENABLED
    mkl_free(chi);
    mkl_free(DM);
    mkl_free(X);
#else
    free(chi);
    free(DM);
    free(X);
#endif
}

void grid::calcDensityScr(){
  atomDensity = 0.0;
  double *lArr, *rArr;
  double hlp[noAOs];

#if MKL_ENABLED
    rArr = (double*)mkl_malloc( noAOs*sizeof(double), 64 );
    lArr = (double*)mkl_malloc( noAOs*sizeof(double), 64 );
#else
    rArr = (double*)malloc( noAOs*sizeof(double) );
    lArr = (double*)malloc( noAOs*sizeof(double) );
#endif 

  for(int p=0; p<noPoints; p++){
    for(int i=0; i<noAOs; i++){lArr[i] = gridValue[p][i];}
    //temporary while cblas.h is not working
    for(int i=0; i<noAOs; i++){
      hlp[i] = 0.0;
      for(int j=0; j<noAOs; j++){
        hlp[i] += densityMatrix[i][j]*gridValue[p][j]; 
      }
      rArr[i] = hlp[i];
    }
    densityScreening(lArr, rArr, p);
    atomDensity += gridDensity[p];    
  }

#if MKL_ENABLED
    //mkl_free(rArr);
    //mkl_free(lArr);
#else
    //free(rArr);
    //free(lArr);
#endif
}

/// protected functions called during calculation
double grid::getValue(int curFnc, double R)
{
    return gridAtom.coef[curFnc]*exp(-1*gridAtom.exp[curFnc]*R);
}

double grid::getR(int curPt, int curShell)
{
  double dx = gridAtom.atomShell[curShell].x - xCoord[curPt];
  double dy = gridAtom.atomShell[curShell].y - yCoord[curPt];
  double dz = gridAtom.atomShell[curShell].z - zCoord[curPt];

  return dx*dx + dy*dy + dz*dz;
}

double grid::getR2(int curPt){
  double dx = xCoord[curPt]*xCoord[curPt];
  double dy = yCoord[curPt]*yCoord[curPt];
  double dz = zCoord[curPt]*zCoord[curPt];

  return dx + dy + dz;
}

int grid::getNoFnc(){
  int noFnc = 0;
  for(int i=0; i<gridAtom.noShl; i++){
    if(gridAtom.atomShell[i].ang == 0){noFnc+=1;}
    if(gridAtom.atomShell[i].ang == 1){noFnc+=3;}
    if(gridAtom.atomShell[i].ang == 2){noFnc+=6;}
    if(gridAtom.atomShell[i].ang == 3){noFnc+=10;}
  }
  if(noFnc != 15)
    return 15;
  return noFnc;
}

void grid::densityScreening(double *arr1, double *arr2, int p){
  int lng = noAOs;
  double precision = 1e-12;
  for(int i=0; i<lng; i++){
    if(arr1[i] < precision && arr1[i] > -precision){
      cleanArr(arr1, i, lng);
      cleanArr(arr2, i, lng);
      i--;
      lng--;  
    }
  }
#if MKL_ENABLED 
    gridDensity[p] = cblas_ddot(lng, arr1, 1, arr2, 1);
#else
    gridDensity[p] = tempArrMull(arr1, arr2, lng);
#endif
  gridDensity[p] *= weight[p];
}

void grid::cleanArr(double *arr, int idx, int lng){
  double hlp;
  if(lng<0){return;};
  for(int i=idx; i<lng; i++){
    hlp = arr[i+1];
    arr[i+1] = arr[i];
    arr[i] = hlp;
  }
  //arr[lng] = NULL;
}

double grid::tempArrMull(double *arr1, double *arr2, int lng){
 double out = 0.0;
 if(lng < 1){return 0.0;}; 

 for(int i=0; i<lng; i++){ 
   out+=arr1[i]*arr2[i];
 }
 return out;
}

double* grid::tempMatMull(double *mat1, double *mat2, int m, int n, int k){
  double *C;
  C = (double*)malloc(m*n*sizeof(double));
  
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      C[i*n+j] = 0.0;
      for(int l=0; l<k; l++){
        C[i*n+j] += mat1[i*k+l]*mat2[l*n+j];
      }
    }
  }
  return C;
}

void grid::vectorwiseProduct(double *A, double *B, int start, int end){
  for(int i=0; i<(end-start); i++){
    gridDensity[start + i] = 0.0;
    for(int j=0; j<noAOs; j++){
      gridDensity[start + i] +=  A[i*noAOs + j]*B[i*noAOs + j];
    }
    gridDensity[start + i] *= weight[start + i];
    atomDensity += gridDensity[start + i];
  }
}
