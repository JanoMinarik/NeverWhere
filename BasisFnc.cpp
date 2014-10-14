#include <iostream>
#include <cmath>
// #include "input.h"

using namespace std;
const char basisOrder[] = {'s','s','p'};
const double tollerance = 1e-10;

// const int numAtoms = 2;
const int gridSize = 10;

// holds information about an atom
struct atom
{
    double Ax, Ay, Az;
    double numBasis;
    double *coeficient;
    double *exponent;
};

// grid for one atom
class grid
{
public:
    void calcGrid();
    void printGrid();
    void setGridCoordinates(double low, double up);
    void evenGridCoordinateX(double min, double max);
    void evenGridCoordinateY(double min, double max);
    void evenGridCoordinateZ(double min, double max);

private:
    atom gridAtom;
    double gridCoordinateX[gridSize]; // store numerical value of coordinate
    double gridCoordinateY[gridSize];
    double gridCoordinateZ[gridSize];
    double gridValue[gridSize][gridSize][gridSize];
    double sType(double rx, double ry, double rz, int curAo);
    double getR(double rx, double ry, double rz);
};

// functions
double grid::getR(double rx, double ry, double rz)
{
    return (rx - gridAtom.Ax)*(rx - gridAtom.Ax) + (ry - gridAtom.Ay)*(ry - gridAtom.Ay) + (rz - gridAtom.Az)*(rz - gridAtom.Az);
}

double grid::sType(double rx, double ry, double rz, int curAo)
{
    return gridAtom.coeficient[curAo]*exp( gridAtom.exponent[curAo]*getR(rx, ry, rz) );
}

{
    double step = (max - min) / gridSize;

    for( int i = 0; i < gridSize; i++ )
    {
        gridCoordinateX[i] = min + i*step;
    }
}

void grid::calcGrid()
{
    for(int i = 0; i < gridSize; i++)
    {
        for(int j = 0; j < gridSize; j++)
        {
            for(int k = 0; k < gridSize; k++)
            {
                gridValue[i][j][k] = 0;
                for(int curAo = 0; curAo < gridAtom.numBasis; curAo++)
                {
                    switch(basisOrder[curAo])
                    {
                    case 's':
                        gridValue[i][j][k] += sType(gridCoordinateX[i], gridCoordinateY[j], gridCoordinateZ[k], curAo);
                        break;
                    }
                }
                if(gridValue[i][j][k] < tollerance)
                    gridValue[i][j][k] = 0;
            }
        }
    }
}

// print grid on screen | file
void grid::printGrid()
{
    double ptValue;

    for(int i = 0; i < gridSize; i++)
    {
        for(int j = 0; j < gridSize; j++)
        {
            ptValue = 0;
            for(int k = 0; k < gridSize; k++)
            {
                ptValue += gridValue[i][j][k];
            }
            cout << gridValue << "\t";
        }
        cout << "\n";
    }
}

// functions for pseudo generation of grid
void grid::evenGridCoordinateX(double min, double max)
{
    double step = (max - min) / gridSize;

    for( int i = 0; i < gridSize; i++ )
    {
        gridCoordinateX[i] = min + i*step;
    }
}

void grid::evenGridCoordinateY(double min, double max)
{
    double step = (max - min) / gridSize;

    for( int i = 0; i < gridSize; i++ )
    {
        gridCoordinateY[i] = min + i*step;
    }
}

void grid::evenGridCoordinateZ(double min, double max)
// testing
int main()
{
    return 1;
}
