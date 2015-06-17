#include <iostream>
#include "ccalc.h"
#include "grid.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define VERBOSE 1

#define IDNX(i,j,ld) (i*ld + j)

void freeMem(double *cd_DM, double *cd_gVal, double *cd_wght, double *cd_gDns){
  cudaFree( cd_DM );
  cudaFree( cd_gVal );
  cudaFree( cd_wght );
  cudaFree( cd_gDns );
}

void freeMemBlas(double *A, double *B, double *C, double *D, double *E){
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(D);
  cudaFree(E);
}

// calculation
__global__ void calcDens(int start, int pts, int noAOs, double *cd_DM, double *cd_gVal, double *cd_wght, double *cd_gDns){
  int tid = blockIdx.x;
  int k, l;

  if( tid + start >= pts ){return;}

  cd_gDns[ start + tid ] = 0.0;  
  for(k=0; k<noAOs; k++){
    for(l=0; l<k; l++){
      cd_gDns[ start + tid ] += 2*cd_DM[ IDNX(k,l,noAOs) ]*cd_gVal[ IDNX(tid+start,k,noAOs) ]*cd_gVal[ IDNX(tid+start,l,noAOs) ];
      }
    cd_gDns[ start + tid ] += cd_DM[ IDNX(k,k,noAOs) ]*cd_gVal[ IDNX(start+tid, k, noAOs) ]*cd_gVal[ IDNX(start+tid, k, noAOs) ];
  }
  cd_gDns[ start + tid ] *= cd_wght[ start + tid ];
}

int calcDensFast( int batchSize, grid *myGrid ){
  //cudaError_t cudaStat;
  cublasStatus_t cblStat;
  cublasHandle_t handle;

  int pts, noAOs;
  clock_t start, end;

  double *batch;
  double *cd_DM, *cd_batch, *cd_X, *cd_wght, *cd_gDns;

  pts = myGrid->noPoints; noAOs = myGrid->noAOs;
 
  batch = (double*)malloc( batchSize*noAOs*sizeof(double) ); 

  cudaMalloc( (void**) &cd_DM, noAOs*noAOs*sizeof(double) );
  cudaMalloc( (void**) &cd_batch, batchSize*noAOs*sizeof(double) );
  cudaMalloc( (void**) &cd_X, batchSize*noAOs*sizeof(double) );
  cudaMalloc( (void**) &cd_wght, batchSize*sizeof(double) );
  cudaMalloc( (void**) &cd_gDns, batchSize*sizeof(double) );

  if(cd_DM == NULL || cd_batch == NULL || cd_X == NULL || cd_wght == NULL || cd_gDns == NULL){
    std::cout << " == ERROR: Device memory allocation failed. ==\n";
    return 1;
  }

  cblStat = cublasCreate(&handle);
  if(cblStat){
    std::cout << " == ERROR: Could not initialize CUBLAS ==\n";
    return 1;
  }

  // start clock, copy data to device and compute batch of points
#if VERBOSE
  std::cout << " Starting electron density computing using CuBlas.\n";
#endif 
  start = clock();
  
  cblStat = cublasSetMatrix(noAOs, noAOs, sizeof(double), myGrid->densityMatrix, noAOs, cd_DM, noAOs);
  if(cblStat){
   std::cout << " == ERROR: Density matrix copying failed.\n";
   return 1;
  }

  for(int i=0; i<pts; i+=batchSize){


  
  }

  if(cblStat){
   std::cout << " == Oops. An computation error occured.\n";
   return 1;
  }

  end = clock();
  myGrid->clcTm += (double)(end - start)/CLOCKS_PER_SEC*1000;

#if VERBOSE
  std::cout << " Density computing using CuBlas completed with SUCCESS.\n";
#endif

  return 0; 
}

// interfaces
void calcDensCuda( int cores, grid *myGrid ){
  int noAOs, noPts;

  noAOs = myGrid->noAOs;
  noPts = myGrid->noPoints;

  clock_t start, end;
  cudaError_t cudaStat;
  double *cd_DM, *cd_gVal, *cd_wght, *cd_gDns; 

#if VERBOSE
  std::cout << " Calculating electron density using CUDA.\n";
#endif
  
  cudaMalloc( (void**) &cd_DM, noAOs*noAOs*sizeof(double) );
  cudaMalloc( (void**) &cd_gVal, noPts*noAOs*sizeof(double) );
  cudaMalloc( (void**) &cd_gDns, noPts*sizeof(double) );
  cudaMalloc( (void**) &cd_wght, noPts*sizeof(double) );

  if( cd_DM == NULL || cd_gVal == NULL || cd_gDns == NULL || cd_wght == NULL ){
    std::cout << " == ERROR CUDA memory allocation failed. ==\n";
    freeMem(cd_DM, cd_gVal, cd_wght, cd_gDns);
    return;
  }
 
  start = clock();
 
  cudaStat = cudaMemcpy( cd_DM, myGrid->densityMatrix, noAOs*noAOs*sizeof(double), cudaMemcpyHostToDevice );
  if(cudaStat){
    std::cout << " Density Matrix data copy failed.\n";
    freeMem(cd_DM, cd_gVal, cd_wght, cd_gDns);
    return;
  }
  
  cudaStat = cudaMemcpy( cd_gVal, (double*)&myGrid->gridValue[0][0], noPts*noAOs*sizeof(double), cudaMemcpyHostToDevice );
  if(cudaStat){
    std::cout << " Grid Values data copy failed.\n";
    freeMem(cd_DM, cd_gVal, cd_wght, cd_gDns);
    return;
  }
  
  cudaStat = cudaMemcpy( cd_wght, myGrid->weight, noPts*sizeof(double), cudaMemcpyHostToDevice );
  if(cudaStat){
    std::cout << " Weights data copy failed.\n";
    freeMem(cd_DM, cd_gVal, cd_wght, cd_gDns);
    return;
  }

  for(int i=0; i<noPts; i+=cores){
    calcDens<<<cores,1>>>(i, noPts, noAOs, cd_DM, cd_gVal, cd_wght, cd_gDns);
  } 

  cudaStat = cudaMemcpy( myGrid->gridDensity, cd_gDns, noPts*sizeof(double), cudaMemcpyDeviceToHost );
  if(cudaStat){
    std::cout << " Memory copy from device to host failed.\n";
    freeMem(cd_DM, cd_gVal, cd_wght, cd_gDns);
    return;
  } 

  myGrid->atomDensity = 0.0;
  for(int i=0; i<noPts; i++){
    myGrid->atomDensity += myGrid->gridDensity[i]; 
  }

  end = clock();
  myGrid->clcTm += (double)(end-start)/CLOCKS_PER_SEC*1000; 
#if VERBOSE
    std::cout << " Electron density calculation using CUDA completed.\n";
#endif

  freeMem(cd_DM, cd_gVal, cd_wght, cd_gDns);
}

