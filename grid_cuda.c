#include <stdio.h>
#include <stdlib.h>
#include <time.h>

noPoints = 10;

/// calc dist from 1st Neon
__global__ void getR( &px, &py, &pz, &ax, &ay, &az, &R ) {
    int tid = blockIdx.x;
    if( tid < noPoints ) {
        R[tid] = ( px[tid] - ax[tid] )*( px[tid] - ax[tid] ) +
                ( py[tid] - ay[tid] )*( py[tid] - ay[tid] ) +
                ( pz[tid] - az[tid] )*( pz[tid] - az[tid] )
    }
}

int main( void ) {
    int noDevices;
    int device;
    cudaDeviceProp prop;
    size_t maxMemory = 0;

    /// get no. of devices
    HANDLE_ERROR( cudaGetDeviceCount( &noDevices ) ) );

    /// get device with max. memory
    for (int i=0; i< noDevices; i++) {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
            ( prop.totalGlobalMem > maxMemory ) {
                maxMemory = prop.totalGlobalMem;
            }
    }
    prop.totalGlobalMem = maxMemory;
    HANDLE_ERROR( cudaChooseDevice( &device, &prop ) );
    HANDLE_ERROR( cudaSetDevice( device ) );

    double px[noPoints], py[noPoints], pz[noPoints], R[noPoints];
    double ax, ay, az;
    int *dev_x, *dev_y, *dev_z, *dev_ax, *dev_ay, *dev_az, *dev_R;
    /// alocate memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_x, noPoints*sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_y, noPoints*sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_z, noPoints*sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_ax, sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_ay, sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_az, sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_R, noPoints*sizeof(double) ) );

    /// fill ax, ay, az with random data
    srand (time(NULL));
    for( int i = 0; i < noPoints; i++ ) {
        px[i] = 10*(double)rand() / RAND_MAX;
        py[i] = 10*(double)rand() / RAND_MAX;
        pz[i] = 10*(double)rand() / RAND_MAX;
    }
    ax = ay = az = 5;

    /// copy the arrays to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_x, px, noPoints*sizeof(double), cudaMemcpyHostToDevice  ) );
    HANDLE_ERROR( cudaMemcpy( dev_y, py, noPoints*sizeof(double), cudaMemcpyHostToDevice  ) );
    HANDLE_ERROR( cudaMemcpy( dev_z, pz, noPoints*sizeof(double), cudaMemcpyHostToDevice  ) );
    HANDLE_ERROR( cudaMemcpy( dev_ax, ax, sizeof(double), cudaMemcpyHostToDevice  ) );
    HANDLE_ERROR( cudaMemcpy( dev_ay, ay, sizeof(double), cudaMemcpyHostToDevice  ) );
    HANDLE_ERROR( cudaMemcpy( dev_az, az, sizeof(double), cudaMemcpyHostToDevice  ) );

    /// do stuff
    getR<<<noPoints,1>>>( dev_x, dev_y, dev_z, dev_ax, dev_ay, dev_az, dev_R );

    /// get results and print
    HANDLE_ERROR( cudaMemcpy( R, dev_R, noPoints*sizeof(double), cudaMemcpyDeviceToHost  ) );
    for( int i = 0; i < noPoints; i++ ) {
        printf("%f \n", R[i]);
    }

    return 0;
}
