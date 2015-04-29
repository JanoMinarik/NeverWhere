#include <iostream>
#include "ccalc.h"
#include "grid.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDNX(i,j,ld) (i*ld + j)

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
        cd_gDns[ start + tid ]+= 2*cd_DM[ IDNX(k,l,noAOs) ]*cd_gVal[ (start+tid)*noAOs + k ]*cd_gVal[ (start+tid)*noAOs + l ];
      }
      cd_gDns[ start + tid ] += cd_DM[ IDNX(k,k,noAOs) ]*cd_gVal[ IDNX(start+tid, k, noAOs) ];
    }
    cd_gDns[ start + tid ] *= cd_wght[ start + tid ];
  }
}

void calcDensFast( int cores, grid myGrid ){
  //cudaError_t cudaStat;
  //cublasStatus_t cblStat;
  cublasHandle_t handle;

  int pts, noAOs;
  double *cd_DM, *cd_gVal, *cd_wght, *cd_gDns;

  pts = myGrid.noPoints; noAOs = myGrid.noAOs;

  cudaMalloc( (void**) &cd_DM, noAOs*noAOs*sizeof(double) );
  cudaMalloc( (void**) &cd_gVal, pts*noAOs*sizeof(double) );
  cudaMalloc( (void**) &cd_wght, pts*sizeof(double) );
  cudaMalloc( (void**) &cd_gDns, pts*sizeof(double) );

  if(cd_DM == NULL || cd_gVal == NULL || cd_wght == NULL || cd_gDns == NULL){
    std::cout << " == Device memory allocation failed. ==\n";
    freeMem(cd_DM, cd_gVal, cd_wght, cd_gDns);
    return;
  }

  cublasCreate(&handle);
  if(handle == NULL){
    std::cout << " == Could not initialize CUBLAS ==\n";
    freeMem(cd_DM, cd_gVal, cd_wght, cd_gDns);
    return;
  }

}

// interfaces
void calcDensCuda( int cores, grid myGrid ){
  cudaError_t cs;
  double *cd_DM, *cd_gVal, *cd_wght, *cd_gDns; 
  
  cudaMalloc( (void**) &cd_DM, myGrid.noAOs*myGrid.noAOs*sizeof(double) );
  cudaMalloc( (void**) &cd_gVal, myGrid.noPoints*myGrid.noAOs*sizeof(double) );
  cudaMalloc( (void**) &cd_gDns, myGrid.noPoints*sizeof(double) );
  cudaMalloc( (void**) &cd_wght, myGrid.noPoints*sizeof(double) );

  if( cd_DM == NULL || cd_gVal == NULL || cd_gDns == NULL || cd_wght == NULL ){
    std::cout << " == ERROR CUDA memory allocation failed. ==\n";
    freeMem(cd_DM, cd_gVal, cd_wght, cd_gDns);
    return;
  }
 
  cs = cudaMemcpy( cd_DM, myGrid.densityMatrix, myGrid.noAOs*myGrid.noAOs*sizeof(double), cudaMemcpyHostToDevice );
  if(cs){std::cout << " Density Matrix data copy failed.\n";}
  double *pGVal = &myGrid.gridValue[0][0];
  cs = cudaMemcpy( cd_gVal, (double*)&myGrid.gridValue[0][0], myGrid.noPoints*myGrid.noAOs*sizeof(double), cudaMemcpyHostToDevice );
  if(cs){std::cout << " Grid Values data copy failed.\n";}
  cs = cudaMemcpy( cd_wght, myGrid.weight, myGrid.noPoints*sizeof(double), cudaMemcpyHostToDevice );
  if(cs){std::cout << " Weights data copy failed.\n";}

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

