#include <iostream>
#include <ctime>
#include "Neon.h"
#include "Grid.h"

using namespace std;

int main()
{
    //Neon myNeon;
    //myNeon.printNeon();
    //myNeon.printShell();

    std::cout << "Grid:\n";
    Grid myGrid;

    ///first scenario - 2 different points
    clock_t start = clock();
    myGrid.setGrid(2);
    myGrid.setCoord();
    myGrid.setCoord(1, 1.5, 1.5, 1.5);
    myGrid.calcGrid();
    clock_t finish = clock();
    myGrid.printGrid();
    cout << "Grid initialization and calculation time: " << (finish-start) << " ns\n";

    ///second scenario - 1 point on the Moon
    start = clock();
    myGrid.setCoord(1, 1500, 1500, 1500);
    myGrid.calcGrid();
    finish = clock();
    myGrid.printGrid();
    cout << "Grid initialization and calculation time: " << (finish-start) << " ns\n";

    myGrid.unsetGrid();

    return 0;
}
