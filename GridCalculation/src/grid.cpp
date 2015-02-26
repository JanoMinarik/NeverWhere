#include "grid.h"
#include "atom.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>

#define PI 3.1415926

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
  std::cout << "no Points: " << noPoints << "\n";
  std::cout << "no AOs: " << noAOs << "\n";
  std::cout << "density: " << atomDensity << "\n";
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

