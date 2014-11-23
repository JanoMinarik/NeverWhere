#include "Neon.h"
#include <iostream>
using namespace std;
double Neon::exponent[25] = {1.788e4, 2.683e3, 6.115e2, 1.735e2, 5.664e1, 2.042e1, 7.81, 1.653, 4.869e-1,
1.788e4, 2.683e3, 6.115e2, 1.735e2, 5.664e1, 2.042e1, 7.81, 1.653, 4.869e-1,
4.869e-1, 2.839e1, 6.27, 1.695, 4.317e-1, 4.317e-1, 2.202};
double Neon::coefficient[25] = {8.13283990e-1, 1.50832394, 2.53133644, 3.69806595, 4.28062152, 3.06933306, 8.59136106e-1, 1.56504324e-2, -8.72387333e-4, -1.89545807e-1,
-3.60541657e-1, -5.90437580e-1, -9.42505698e-1, -1.12138075, -1.19964549, -3.56398904e-4,
5.89163852e-1, 2.34803397e-1, 4.15422412e-1, 4.30502365, 3.39674885, 1.40249651, 2.27278748e-1,
4.98789898e-1, 1.13475050e1 };

Neon::Neon()
{
    int ctr = 1;
    // initialize shell number
    for ( int i = 0; i < 25; i++)
    {
        if( i < 9 )
            shellNumber[i] = 1;
        else if( i < 18 )
            shellNumber[i] = 2;
        else if( i == 18 )
            shellNumber[i] = 3;
        else if( i < 23 )
            shellNumber[i] = 4;
        else if( i == 23 )
            shellNumber[i] = 5;
        else
            shellNumber[i] = 6;
    }
// initialize function number
    for ( int i = 0; i < 25; i++)
    {
        if( (i > 1) && (shellNumber[i] != shellNumber[i-1]) )
            ctr = 1;
        shellFunction[i] = ctr++;
    }

    int ang;
    for(int i = 0; i < 6; i++)
    {
        if(i < 3)
            ang = 0;
        else if(i < 5)
            ang = 1;
        else
            ang = 2;
        setShell(i, ang, 0, 0, 0);
    }
}

void Neon::printNeon()
{
    cout << "shell\tfunction\texponent\tcontraction coefficient"<< "\n";
    for ( int i = 0; i < 25; i++ )
    {
        cout << shellNumber[i] << "\t";
        cout << shellFunction[i] << "\t\t";
        cout << exponent[i] << "\t\t";
        cout << coefficient[i] << "\n";
    }
}

void Neon::printShell()
{
    std::cout << "shell \n";
    for( int i = 0; i < 6; i++ )
    {
        std::cout << i+1 << "\t" << shellCord[i].ang << "\t" << shellCord[i].x << "\t" << shellCord[i].y << "\t" << shellCord[i].z << "\n";
    }
}

void Neon::setShell(int curShell, int sAng, double sx, double sy, double sz)
{
    shellCord[curShell].ang = sAng;
    shellCord[curShell].x = sx;
    shellCord[curShell].y = sy;
    shellCord[curShell].z = sz;
}
