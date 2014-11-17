#include "grid.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void Grid::setGrid(int numN, int numP)
{
    numNeons = numN;
    numPoints = numP;
    numFnc = getNumFnc();
    gridNeons = new Neon[numNeons];
    xCoord = new double[numPoints];
    yCoord = new double[numPoints];
    zCoord = new double[numPoints];
    gridValue = new double*[numPoints];

    for( int i = 0; i < numPoints; i++ )
    {
        gridValue[i] = new double[numFnc];
    }
    std::cout << "Initialization succeed\n";
}

void Grid::unsetGrid()
{
    delete [] xCoord;
    delete [] yCoord;
    delete [] zCoord;
    delete [] gridNeons;

    for( int i = 0; i < numPoints; i++ )
    {
        delete [] gridValue[i];
    }
    delete gridValue;
}

int Grid::getNumFnc() {
    int count = 0;

    for( int i = 0; i < 6; i++ )
    {
        if( gridNeons[0].shellCord[i].ang == 0 )
            count += 1;
        if( gridNeons[0].shellCord[i].ang == 1 )
            count += 3;
        if( gridNeons[0].shellCord[i].ang == 2 )
            count += 6;
    }

    return count;
}
// set grid Atoms
void Grid::setNeon(int curAt, double x, double y, double z)
{

}
// generate coordinates randomly
void Grid::setCoord(int range)
{
    for( int i = 0; i < numPoints; i++ )
    {
        xCoord[i] = range*(double)rand() / RAND_MAX;
        yCoord[i] = range*(double)rand() / RAND_MAX;
        zCoord[i] = range*(double)rand() / RAND_MAX;
    }
}
// set coordinates manualy
void Grid::setCoord(int curPt, double x, double y, double z)
{
    xCoord[curPt] = x;
    yCoord[curPt] = y;
    zCoord[curPt] = z;
}
// calculate energy at given points
void Grid::calcGrid()
{
    int curValue, curFnc;
    double value, R;
    for( int i = 0; i < numPoints; i++ )
    {
        curValue = curFnc = 0;
        for( int j = 0; j < 6; j++ )
        {
            R = getR(i, j);
            value = 0;
            while( shellNumber[curFnc] == (j+1) )
                {
                    value += getValue(curFnc, R);
                    curFnc++;
                }

            if( shellCord[j].ang == 0 )
            {
                gridValue[i][curValue++] = value;
            }

            if( shellCord[j].ang == 1 )
            {
                gridValue[i][curValue++] = xCoord[i]*value;
                gridValue[i][curValue++] = yCoord[i]*value;
                gridValue[i][curValue++] = zCoord[i]*value;
            }

            if( shellCord[j].ang == 2 )
            {
                gridValue[i][curValue++] = xCoord[i]*xCoord[i]*value;
                gridValue[i][curValue++] = yCoord[i]*yCoord[i]*value;
                gridValue[i][curValue++] = zCoord[i]*zCoord[i]*value;
                gridValue[i][curValue++] = xCoord[i]*yCoord[i]*value;
                gridValue[i][curValue++] = xCoord[i]*zCoord[i]*value;
                gridValue[i][curValue++] = yCoord[i]*zCoord[i]*value;
            }
        }
    }
}

double Grid::getValue(int curAt, int curFnc, double R)
{
    return gridNeons[curAt].coefficient[curFnc]*exp(-1*gridNeons[curAt].exponent[curFnc]*R);
}

double Grid::getR(int curAt, int curPt, int curShell)
{
    return (xCoord[curPt] - gridNeons[curAt].shellCord[curShell].x)*(xCoord[curPt] - gridNeons[curAt].shellCord[curShell].x) +
    (yCoord[curPt] - gridNeons[curAt].shellCord[curShell].y)*(yCoord[curPt] - gridNeons[curAt].shellCord[curShell].y) +
    (zCoord[curPt] - gridNeons[curAt].shellCord[curShell].z)*(zCoord[curPt] - gridNeons[curAt].shellCord[curShell].z);
}
// print
void Grid::printGrid()
{
    std::cout << "Contracted Functions: " << numFnc << "\n";
    std::cout << "Points: " << numPoints << "\n";
    for( int i = 0; i < numPoints; i++ )
    {
        std::cout << "Point " << i+1 << "\n" << "Coordinates: \n";
        std::cout << xCoord[i] << "\t" << yCoord[i] << "\t" << zCoord[i] << "\nContracted Functions:\n";
        for( int j = 0; j < numFnc; j++ )
        {
            std::cout << gridValue[i][j] << "\n";
        }
    }
}
