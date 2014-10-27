#include "grid.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

void Grid::setGrid(int num)
{
    numPoints = num;
    numFnc = getNumFnc();
    xCoord = new double[num];
    yCoord = new double[num];
    zCoord = new double[num];
    gridValue = new double*[num];

    for( int i = 0; i < num; i++ )
    {
        gridValue[i] = new double[numFnc];
    }
    std::cout << "Initialization succeed";
}

void Grid::unsetGrid()
{
    delete [] xCoord;
    delete [] yCoord;
    delete [] zCoord;

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
        if( shellCord[i].ang == 0 )
            count += 1;
        if( shellCord[i].ang == 1 )
            count += 3;
        if( shellCord[i].ang == 2 )
            count += 6;
    }
}
// generate coordinates randomly
void Grid::setCoord()
{
    for( int i = 0; i < numPoints; i++ )
    {
        xCoord[i] = (double)rand() / RAND_MAX;
        yCoord[i] = (double)rand() / RAND_MAX;
        zCoord[i] = (double)rand() / RAND_MAX;
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
    int curValue;
    for( int i = 0; i < numPoints; i++ )
    {
        curValue = 0;
        for( int j = 0; j < 6; j++ )
        {
            if( shellCord[j].ang == 0 )
                gridValue[i][curValue++] += getValue(i, j);
            if( shellCord[j].ang == 1 )
            {
                gridValue[i][curValue++] += xCoord[i]*getValue(i, j);
                gridValue[i][curValue++] += yCoord[i]*getValue(i, j);
                gridValue[i][curValue++] += zCoord[i]*getValue(i, j);
            }
        }
    }
}

double Grid::getValue(int curPt, int curShell)
{

}
// print
void Grid::printGrid()
{
    for( int i = 0; i < numPoints; i++ )
    {
        std::cout << xCoord[i] << "\t" << yCoord[i] << "\t" << zCoord[i] << "\t";
        for( int j = 0; j < numFnc; j++ )
        {
            std::cout << gridValue[i][j] << "\t";
        }
        std::cout << "\n";
    }
}
