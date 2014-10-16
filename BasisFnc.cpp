#include <iostream>
#include <iomanip>
#include <cmath>
// my libraries
// #include <grid.h>
// #include <input.h>

using namespace std;
const double tollerance = 1e-10;
unsigned int gridSize = 10;

// holds information about an atom
struct atom
{
    double Ax, Ay, Az;
    int numBasis;
    char *basisTypes;
    double *coeficient; // basis koeficient
    double *exponent; // basis exponent
};

// grid for one atom
class grid
{
public:
    atom gridAtom;
    grid();
    ~grid();
    void calcGrid();
    void printGrid2D();
    void evenGridCoordinateX(double min, double max);   // only for testing
    void evenGridCoordinateY(double min, double max);
    void evenGridCoordinateZ(double min, double max);

private:
    double *gridCoordinateX; // store numerical value of coordinate
    double *gridCoordinateY;
    double *gridCoordinateZ;
    double ***gridValue;
    double sType(double rx, double ry, double rz, int curAo);
    double getR(double rx, double ry, double rz);
};

// constructor & deconstructor
grid::grid()
{
    gridCoordinateX = new double[gridSize];
    gridCoordinateY = new double[gridSize];
    gridCoordinateZ = new double[gridSize];
    gridValue = new double**[gridSize];
    for( int i = 0; i < gridSize; i++ )
    {
        gridValue[i] = new double*[gridSize];
        for( int j = 0; j < gridSize; j++ )
        {
            gridValue[i][j] = new double[gridSize];
        }
    }
}

grid::~grid()
{
    delete [] gridCoordinateX;
    delete [] gridCoordinateY;
    delete [] gridCoordinateZ;

    for (int i = 0; i < gridSize; i++)
    {
        for (int j = 0; j < gridSize; j++)
        {
            delete [] gridValue[i][j];
        }
        delete [] gridValue[i];
    }
    delete [] gridValue;
}

// functions
double grid::getR(double rx, double ry, double rz)
{
    return ((rx - gridAtom.Ax)*(rx - gridAtom.Ax)) + ((ry - gridAtom.Ay)*(ry - gridAtom.Ay)) + ((rz - gridAtom.Az)*(rz - gridAtom.Az));
}

double grid::sType(double rx, double ry, double rz, int curAo)
{
    return gridAtom.coeficient[curAo]*exp( -1.0*gridAtom.exponent[curAo]*getR(rx, ry, rz) );
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
                    switch(gridAtom.basisTypes[curAo])
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
void grid::printGrid2D()
{
    double ptValue;

    for(int i = 0; i < gridSize; i++)
    {
        cout << setprecision(3) << gridCoordinateY[i] << ":\t";
        for(int j = 0; j < gridSize; j++)
        {
            ptValue = 0;
            for(int k = 0; k < 1; k++)
            {
                ptValue += gridValue[i][j][k];
            }
            cout << setprecision(5) << ptValue << "\t";
        }
        cout << "\n";
    }

    cout << "y/x\t";
    for(int j = 0; j < gridSize; j++)
        cout << setprecision(3) << gridCoordinateX[j] << "\t";
    cout << "\n";
}

// functions for pseudo generation of grid
void grid::evenGridCoordinateX(double min, double max)
{
    double step = (max - min) / (gridSize - 1);

    for( int i = 0; i < gridSize; i++ )
    {
        gridCoordinateX[i] = min + i*step;
    }
}

void grid::evenGridCoordinateY(double min, double max)
{
    double step = (max - min) / (gridSize - 1);

    for( int i = 0; i < gridSize; i++ )
    {
        gridCoordinateY[i] = min + i*step;
    }
}

void grid::evenGridCoordinateZ(double min, double max)
{
    double step = (max - min) / (gridSize - 1);

    for( int i = 0; i < gridSize; i++ )
    {
        gridCoordinateZ[i] = min + i*step;
    }
}

// test scenario, H - 3=STO-3G basis set
int main()
{
    grid myGrid;
    // input data
    myGrid.gridAtom.Ax = 0.4;
    myGrid.gridAtom.Ay = 0.5;
    myGrid.gridAtom.Az = 0.6;
    myGrid.gridAtom.numBasis = 1;
    myGrid.gridAtom.coeficient = new double[3];
    myGrid.gridAtom.exponent = new double[3];
    myGrid.gridAtom.basisTypes = new char[3];
    myGrid.gridAtom.coeficient[0] = 0.44;
    myGrid.gridAtom.coeficient[1] = 0.53;
    myGrid.gridAtom.coeficient[2] = 0.15;
    myGrid.gridAtom.exponent[0] = 0.11;
    myGrid.gridAtom.exponent[1] = 0.41;
    myGrid.gridAtom.exponent[2] = 2.23;
    myGrid.gridAtom.basisTypes[0] = 's';
    myGrid.gridAtom.basisTypes[1] = 's';
    myGrid.gridAtom.basisTypes[2] = 's';
    myGrid.evenGridCoordinateX(0.1, 1.5);
    myGrid.evenGridCoordinateY(0.1, 1.5);
    myGrid.evenGridCoordinateZ(0.1, 1.5);
    // calculation
    myGrid.calcGrid();
    myGrid.printGrid2D();

    return 1;
}
