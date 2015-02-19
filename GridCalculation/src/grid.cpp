#include "grid.h"
#include "atom.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/// grid memory alocation/ free.
void grid::setGrid(int noP)
{
    noPoints = noP;
    xCoord = new double[noPoints];
    yCoord = new double[noPoints];
    zCoord = new double[noPoints];
    gridDensity = new double[noPoints];
    gridValue = new double*[noPoints];
    int noAOs = getNoFnc();
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

void grid::setDensityMatrix(double arr[15][15]){
  int n = getNoFnc();
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      densityMatrix[i][j] = arr[i][j];
    }
  }
}

void grid::setCoord(double range){
    for( int i = 0; i < noPoints; i++ )
    {
        xCoord[i] = range*(double)rand() / RAND_MAX;
        yCoord[i] = range*(double)rand() / RAND_MAX;
        zCoord[i] = range*(double)rand() / RAND_MAX;
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

    for( int i=0; i<noPoints; i++ )
    {
        xCoord[i] = start + i*step;
        yCoord[i] = start + i*step;
        zCoord[i] = start + i*step;
    }
}

void grid::setCoord(int curPt, double x, double y, double z)
{
    xCoord[curPt] = x;
    yCoord[curPt] = y;
    zCoord[curPt] = z;
}

/// print grid
void grid::printGrid(){
  for(int i=0; i<noPoints; i++){
    std::cout << xCoord[i] << " " << yCoord[i] << " " << zCoord[i] << " ";
    std::cout << gridDensity[i] << "\n";
  }
}

void grid::printDensityMatrix(){
  for(int i=0; i<getNoFnc(); i++){
    for(int j=0; j<getNoFnc(); j++){
      std::cout << densityMatrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}

void grid::printGridInfo(){
  std::cout << "no Points: " << noPoints << "\n";
  std::cout << "no AOs: " << getNoFnc() << "\n";
}

void grid::printFullGrid(){
  for(int i=0; i<noPoints; i++){
    std::cout << xCoord[i] << " " << yCoord[i] << " " << zCoord[i] << " ";
    for(int j=0; j<getNoFnc(); j++){
      std::cout << gridValue[i][j] << " ";
    }
    std::cout << gridDensity[i] << "\n";
  }
}

/// calculate grid density
void grid::calcGrid()
{
    int curValue, curFnc;
    double value, R;
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
    calcDensity();
}

void grid::calcDensity(){
  for(int p=0; p<noPoints; p++){
    gridDensity[p] = 0.0;
    for(int k=0; k<15; k++){
      for(int l=0; l<k; l++){
        gridDensity[p] += 2*densityMatrix[k][l]*gridValue[p][k]*gridValue[p][l];
      }
      gridDensity[p] += densityMatrix[k][k]*gridValue[p][k]*gridValue[p][k];
    }
  }
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

int grid::getNoFnc(){
  int noFnc = 0;
  //for(int i=0; i<gridAtom.noShl; i++){
  //  if(gridAtom.atomShell[i].ang == 0){noFnc+=1;}
  //  if(gridAtom.atomShell[i].ang == 1){noFnc+=3;}
  //  if(gridAtom.atomShell[i].ang == 2){noFnc+=6;}
  //  if(gridAtom.atomShell[i].ang == 3){noFnc+=10;}
  //}
  return 15;
  //return noFnc;
}

