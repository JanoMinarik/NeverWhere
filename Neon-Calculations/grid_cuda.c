#include <stdio.h>

/// grid manipulations functions

void initGrid( gridCuda curGrid, int noPts, int noAts ) {

    curGrid.numNeons = noAts;
    curGrid.numPoints = noPts;
    curGrid.gridNeons = malloc( curGrid.numNeons*sizeof( Neon ) );
    curGrid.numFnc = getNumFnc( curGrid.gridNeons[0] );
    curGrid.xCoord = malloc( curGrid.numPoints*sizeof( double ) );
    curGrid.yCoord = malloc( curGrid.numPoints*sizeof( double ) );
    curGrid.zCoord = malloc( curGrid.numPoints*sizeof( double ) );
    curGrid.gridValue = (*double)malloc( curGrid.numPoints*sizeof( double ) );

    for( int i=0; i<curGrid.numPoints; i++ ) {
        curGrid.gridValue[i] = malloc( curGrid.numFnc*sizeof( double ) );
    }

    printf("Grid initialization completed.\n");
}

void calculateGrid( gridCuda curGrid ) {
}

void printGrid( ) {
}

/// kernel functions

/// supporting functions

int getNumFnc( Neon myNeon ) {
    int count = 0;

    for( int i = 0; i < 6; i++ )
    {
        if( myNeon.shellCord[i].ang == 0 )
            count += 1;
        if( myNeon.shellCord[i].ang == 1 )
            count += 3;
        if( myNeon.shellCord[i].ang == 2 )
            count += 6;
    }

    return count;
}

void initDevices(  ) {
}

void freeDevices(  ) {
}

void host2device( ) {
}

void device2host( ) {
}
