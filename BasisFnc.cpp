#include <iostream>
#include <cmath>

const int gridSize = 20;
const int numAtoms = 3;
const double tollerance = 1e-6;
using namespace std;

// structure for a Atom
struct atom
{
    int rxA, ryA, rzA;
    double coef;
    double exp;
};

// base class for sType base function
class grid
{
    public:
        void setAtomicCoordinates(int[], int[], int[], double[], double[] );
        void calcEnergy();
        void printGrid();

    private:
        atom Atoms[numAtoms];
        double gridEnergy[gridSize][gridSize][gridSize];
        double sType(int rx, int ry, int rz, int curAtom);
        double getR(int rx, int ry, int rz, int rxA, int ryA, int rzA);
};

// set parameters for atoms
void grid::setAtomicCoordinates(int x[], int y[], int z[], double coef[], double exp[])
{
    for(int i = 0; i < numAtoms; i++)
    {
        Atoms[i].rxA = x[i];
        Atoms[i].ryA = y[i];
        Atoms[i].rzA = z[i];
        Atoms[i].coef = coef[i];
        Atoms[i].exp = exp[i];
    }
}

void grid::calcEnergy()
{
    int curAtom;

    for(int i = 0; i < gridSize; i++)
    {
        for(int j = 0; j < gridSize; j++)
        {
            for(int k = 0; k < gridSize; k++)
            {
                gridEnergy[i][j][k] = 0;
                for(curAtom = 0; curAtom < numAtoms; curAtom++)
                {
                    gridEnergy[i][j][k] += sType(i, j, k, curAtom);
                }
                if(gridEnergy[i][j][k] < tollerance)
                    gridEnergy[i][j][k] = 0;
            }
        }
    }

    cout << "Calculation completed with success.\n";
}
// calculate Energy from a single atom
double grid::sType(int rx, int ry, int rz, int curAtom)
{
    return Atoms[curAtom].coef * exp(-Atoms[curAtom].exp*getR(rx, ry, rz, Atoms[curAtom].rxA, Atoms[curAtom].ryA, Atoms[curAtom].rzA));
}
// calculates distance
double grid::getR(int rx, int ry, int rz, int rxA, int ryA, int rzA)
{
    return (rx - rxA)*(rx - rxA) + (ry - ryA)*(ry - ryA) + (rz - rzA)*(rz - rzA);
}

void grid::printGrid()
{
    double gridValue;

    for(int i = 0; i < gridSize; i++)
    {
        for(int j = 0; j < gridSize; j++)
        {
            gridValue = 0;
            for(int k = 0; k < gridSize; k++)
            {
                gridValue += gridEnergy[i][j][k];
            }
            cout << gridValue << " ";
        }
        cout << "\n";
    }
}

// main for calculation testing
int main()
{
    grid SmallField;
    int x[] = {0, 5, 2};
    int y[] = {0, 0, 5};
    int z[] = {0, 5, 0};
    double c[] = {0.6, 0.2, 0.2};
    double alfa[] = {10, 1, 1};
    SmallField.setAtomicCoordinates(x, y, z, c, alfa);
    SmallField.calcEnergy();
    SmallField.printGrid();

    return 0;
}
