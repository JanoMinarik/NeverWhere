#include "grid.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

void Grid::setGrid(int num)
{
    numPoints = num;
    numFnc = 15;
    xCoord = new double[num];
    yCoord = new double[num];
    zCoord = new double[num];
    gridValue = new double*[num];

    for( int i = 0; i < num; i++ )
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

    return count;
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
    int curValue, curFnc;
    double value, R;
    for( int i = 0; i < numPoints; i++ )
    {
        curValue = curFnc = 0;
        for( int j = 0; j < 6; j++ )
        {
            R = getR(i, j);
            value = 0;

            if( shellCord[j].ang == 0 )
            {
                while( shellNumber[curFnc] == (j+1) )
                {
                    value += getValue(curFnc, R);
                    curFnc++;
                }
                gridValue[i][curValue++] = value;
            }

            if( shellCord[j].ang == 1 )
            {
                while( shellNumber[curFnc] == (j+1) )
                {
                    value += getValue(curFnc, R);
                    curFnc++;
                }
                gridValue[i][curValue++] = xCoord[i]*value;
                gridValue[i][curValue++] = yCoord[i]*value;
                gridValue[i][curValue++] = zCoord[i]*value;
            }

            if( shellCord[j].ang == 2 )
            {
                gridValue[i][curValue++] = 10;
                gridValue[i][curValue++] = 10;
                gridValue[i][curValue++] = 10;
                gridValue[i][curValue++] = 10;
                gridValue[i][curValue++] = 10;
                gridValue[i][curValue++] = 10;
            }
        }
    }
}

double Grid::getValue(int curFnc, double R)
{
    return 1;
}

double Grid::getR(int curPt, int curShell)
{
    return (xCoord[curPt] - shellCord[curShell].x)*(xCoord[curPt] - shellCord[curShell].x) +
    (yCoord[curPt] - shellCord[curShell].y)*(yCoord[curPt] - shellCord[curShell].y) +
    (zCoord[curPt] - shellCord[curShell].z)*(zCoord[curPt] - shellCord[curShell].z);
}
// print
void Grid::printGrid()
{
    std::cout << "Contracted Functions: " << numFnc << "\n";
    std::cout << "Points: " << numPoints << "\n";
    for( int i = 0; i < numPoints; i++ )
    {
        std::cout << "Point " << i+1 << "\n";
        std::cout << xCoord[i] << "\t" << yCoord[i] << "\t" << zCoord[i] << "\t";
        for( int j = 0; j < numFnc; j++ )
        {
            std::cout << gridValue[i][j] << "\t";
        }
        std::cout << "\n";
    }
}
