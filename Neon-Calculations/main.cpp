#include <iostream>
#include <math.h>
#include "Neon.h"
#include "Grid.h"

using namespace std;

int main()
{
    Neon myNeon;
    myNeon.printNeon();

    std::cout << "Grid:\n";
    Grid myGrid;
    myGrid.setGrid(2);
    myGrid.setCoord();
    //myGrid.setCoord(2, 1.5, 1.5, 1.5);
    myGrid.calcGrid();
    myGrid.printGrid();
    myGrid.unsetGrid();

    return 0;
}
