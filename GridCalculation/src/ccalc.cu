#include <iostream>
#include "ccalc.h"
#include "grid.h"

void freeMem(double *cd_DM, double *cd_gVal, double *cd_wght, double *cd_gDns){
  cudaFree( cd_DM );
  cudaFree( cd_gVal );
  cudaFree( cd_wght );
  cudaFree( cd_gDns );
}

// calculation
__global__ void calcDens(int start, int pts, int noAOs, double *cd_DM, double *cd_gVal, double *cd_wght, double *cd_gDns){
  int tid = blockIdx.x;
  int k, l;

  if( tid + start < pts ){
    cd_gDns[ start + tid ] = 0.0;  
    for(k=0; k<noAOs; k++){
      for(l=0; l<k; l++){
        cd_gDns[ start + tid ]+= 2*cd_gDns[ (start+tid)*noAOs + l ]*cd_gVal[ (start+tid)*noAOs + k ]*cd_gVal[ (start+tid)*noAOs + l ];
      }
      cd_gDns[ start + tid ] += cd_gDns[ (start+tid)*noAOs ];
    }
    cd_gDns[ start + tid ] *= cd_wght[ start + tid ];
  }
}

__global__ void calcDensFast(int start, int pts, int noAOs, double *cd_DM, double *cd_gVal, double *cd_wght, double *cd_gDns){

}

// interfaces
void calcDensCuda(int opt, int cores, grid myGrid){
  double *cd_DM, *cd_gVal, *cd_wght, *cd_gDns; 
  
  cudaMalloc( (void**)cd_DM, myGrid.noAOs*myGrid.noAOs*sizeof(double) );
  cudaMalloc( (void**)cd_gVal, myGrid.noPoints*myGrid.noAOs*sizeof(double) );
  cudaMalloc( (void**)cd_gDns, myGrid.noPoints*sizeof(double) );
  cudaMalloc( (void**)cd_wght, myGrid.noPoints*sizeof(double) );

  if( cd_DM == NULL || cd_gVal == NULL || cd_gDns == NULL || cd_wght == NULL ){
    std::cout << " == ERROR CUDA memory allocation failed. ==\n";
    return;
  }
 
  std::cout << "Size of cd_DM: " << sizeof(cd_DM)  << "\n";
  std::cout << "Size of cd_gVal: " << sizeof(cd_gVal)  << "\n";
  std::cout << "Size of cd_gDns: " << sizeof(cd_gDns)  << "\n";
  std::cout << "Size of cd_wght: " << sizeof(cd_wght)  << "\n";
 
  cudaMemcpy( cd_DM, myGrid.densityMatrix, myGrid.noAOs*myGrid.noAOs*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( cd_gVal, myGrid.gridValue, myGrid.noPoints*myGrid.noAOs*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( cd_wght, myGrid.weight, myGrid.noPoints*sizeof(double), cudaMemcpyHostToDevice );

  for(int i=0; i<myGrid.noPoints; i+=cores){
    calcDens<<<cores,1>>>(i, cores, myGrid.noAOs, cd_DM, cd_gVal, cd_wght, cd_gDns);
  } 

  cudaMemcpy( myGrid.gridDensity, cd_gDns, myGrid.noPoints*sizeof(double), cudaMemcpyDeviceToHost );
  myGrid.atomDensity = 0.0;
  for(int i=0; i<myGrid.noPoints; i++){
    myGrid.atomDensity += myGrid.gridDensity[i]; 
  }

  freeMem(cd_DM, cd_gVal, cd_wght, cd_gDns);
}

