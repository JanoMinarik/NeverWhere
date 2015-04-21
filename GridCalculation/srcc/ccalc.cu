#include <iostream>
#include "ccalc.h"
#include "grid.h"

// data manipulation
void cpyInitial(double *DM, double *cd_DM, double *gridValue, double *cd_gVal,
                           double *weight, double *cd_wght, double *cd_gDns,
                           int pts, int noAOs ){

  cudaMalloc( (void**)cd_DM, noAOs*noAOs*sizeof(double) );
  cudaMalloc( (void**)cd_gVal, pts*noAOs*sizeof(double) );
  cudaMalloc( (void**)cd_gDns, pts*sizeof(double) );
  cudaMalloc( (void**)cd_wght, pts*sizeof(double) );

  if( cd_DM == NULL || cd_gVal == NULL || cd_gDns == NULL || cd_wght == NULL ){
    std::cout << " == ERROR CUDA memory allocation failed. ==\n";
    return;
  }

  cudaMemcpy( cd_DM, DM, pts*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( cd_gVal, gridValue, pts*noAOs*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( cd_wght, weight, pts*sizeof(double), cudaMemcpyHostToDevice );
}

void cpyResult(double *gridDensity, double *cd_gDns, int pts){
  cudaMemcpy( gridDensity, cd_gDns, pts*sizeof(double), cudaMemcpyDeviceToHost );


}

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
  cd_DM = cd_gVal = cd_wght = cd_gDns = NULL;  

  cpyInitial(&myGrid.densityMatrix[0][0], cd_DM, &myGrid.gridValue[0][0], cd_gVal, myGrid.weight, cd_wght, 
             cd_gDns, myGrid.noPoints, myGrid.noAOs);
 
  for(int i=0; i<myGrid.noPoints; i+=cores){
    calcDens<<<cores,1>>>(i, cores, myGrid.noAOs, cd_DM, cd_gVal, cd_wght, cd_gDns);
  } 

  cpyResult(myGrid.gridDensity, cd_gDns, myGrid.noPoints);
  myGrid.atomDensity = 0.0;
  for(int i=0; i<myGrid.noPoints; i++){
    myGrid.atomDensity += myGrid.gridDensity[i]; 
  }
}

