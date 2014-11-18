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
    if( count < 15 ) count = 15;
    return count;
}
/// set grid Atoms
void Grid::setNeon(int curAt, double x, double y, double z)
{
    int ang;

    for( int i = 0; i < 6; i++ )
    {
        if(i < 3)
            ang = 0;
        else if(i < 5)
            ang = 1;
        else
            ang = 2;
        gridNeons[curAt].setShell(i, ang, x, y, z);
    }
}
/// generate coordinates randomly
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
/// calculate energy at given points
void Grid::calcGrid()
{
    int curValue, curFnc;
    double value, R;
    for( int curPt = 0; curPt < numPoints; curPt++ )
    {
        for( int curAt = 0; curAt < numNeons; curAt++ )
        {
            curValue = curFnc = 0;
            for( int j = 0; j < 6; j++ ) /// j =  curShell
            {
                R = getR(curAt, curPt, j);
                value = 0;
                while( gridNeons[curAt].shellNumber[curFnc] == (j+1) )
                {
                    value += getValue(curAt, curFnc, R);
                    curFnc++;
                }

                if( gridNeons[curAt].shellCord[j].ang == 0 )
                {
                    if( curAt == 0 ){ gridValue[curPt][curValue++] = value;}
                    else{ gridValue[curPt][curValue++] += value; }
                }

                if( gridNeons[curAt].shellCord[j].ang == 1 )
                {
                    if( curAt == 0 ){
                        gridValue[curPt][curValue++] = xCoord[curPt]*value;
                        gridValue[curPt][curValue++] = yCoord[curPt]*value;
                        gridValue[curPt][curValue++] = zCoord[curPt]*value;
                    }else{
                        gridValue[curPt][curValue++] += xCoord[curPt]*value;
                        gridValue[curPt][curValue++] += yCoord[curPt]*value;
                        gridValue[curPt][curValue++] += zCoord[curPt]*value;
                    }
                }

                if( gridNeons[curAt].shellCord[j].ang == 2 )
                {
                    if( curAt == 0 ){
                        gridValue[curPt][curValue++] = xCoord[curPt]*xCoord[curPt]*value;
                        gridValue[curPt][curValue++] = yCoord[curPt]*yCoord[curPt]*value;
                        gridValue[curPt][curValue++] = zCoord[curPt]*zCoord[curPt]*value;
                        gridValue[curPt][curValue++] = xCoord[curPt]*yCoord[curPt]*value;
                        gridValue[curPt][curValue++] = xCoord[curPt]*zCoord[curPt]*value;
                        gridValue[curPt][curValue++] = yCoord[curPt]*zCoord[curPt]*value;
                    }else{
                        gridValue[curPt][curValue++] += xCoord[curPt]*xCoord[curPt]*value;
                        gridValue[curPt][curValue++] += yCoord[curPt]*yCoord[curPt]*value;
                        gridValue[curPt][curValue++] += zCoord[curPt]*zCoord[curPt]*value;
                        gridValue[curPt][curValue++] += xCoord[curPt]*yCoord[curPt]*value;
                        gridValue[curPt][curValue++] += xCoord[curPt]*zCoord[curPt]*value;
                        gridValue[curPt][curValue++] += yCoord[curPt]*zCoord[curPt]*value;
                    }
                }
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
/// print
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
