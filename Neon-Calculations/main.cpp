#include <iostream>
#include <ctime>
//#include "neon.h"
#include "grid.h"

using namespace std;

int main()
{
    /// Neon for testing
    /*Neon myNeon;
    myNeon.printNeon();
    myNeon.printShell();*/

    std::cout << "Grid:\n";
    Grid myGrid;
    ///first scenario - 2 different Neons in grid
    /// pt 9 - longest distance, pt 10 - on the moon
    clock_t start = clock();
    myGrid.setGrid(2, 10);
    myGrid.setCoord(2);
    myGrid.setNeon(1, 0, 0, 4);
    myGrid.setCoord(8, 5, 5, 5);
    myGrid.setCoord(9, 1000, 1000, 1000);
    myGrid.calcGrid();
    clock_t finish = clock();

    myGrid.printGrid();
    cout << "Grid initialization and calculation time: " << (finish-start) << " ms\n";
    myGrid.unsetGrid();

    return 0;
}
