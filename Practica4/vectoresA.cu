// Suma de vectores secuencial
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <helper_cuda.h>
#include <helper_timer.h>

StopWatchInterface *hTimer = NULL;
StopWatchInterface *kTimer = NULL;

typedef int *vector;



void LoadP(vector P, unsigned int n)
{
   unsigned int i;

   for (i=0;i<n;i++) 
     P[i] = i;
}

// Function for generating random values for a vector
void LoadStartValuesIntoVectorRand(vector V, unsigned int n)
{
   unsigned int i;

   for (i=0;i<n;i++) 
     V[i] = (int)(random()%9);
}


// Function for printing a vector
void PrintVector(vector V, unsigned int n)
{
   unsigned int i;

   for (i=0;i<n;i++)
      printf("%d\n",V[i]);
}

// Suma vectores cC = cA + cB
__global__ void SumVectorCuda(vector cA, vector cB, vector cC, vector cP, unsigned int n, unsigned int v)
{
   unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
   int end = ((idx + 1) * v) - 1;
   int tid /*= idx * v*/;
   //printf("End: %d\n", end);
   
   for (tid = idx * v; tid <= end ; tid++){
      //printf("Tid: %d, Thread: %d\n", tid, idx);
      //printf("Vector cA: %d Thread: %d\n", cA[tid], idx);
      cC[cP[tid]] = cA[cP[tid]] + cB[cP[tid]];
   }
}


// ------------------------
// MAIN function
// ------------------------
int main(int argc, char **argv)
{
   float timerValue;
   double ops;
   unsigned int n, v;

   //Pasar numero de componentes del vector por thread (v).
   if (argc == 3){
      n = atoi(argv[1]);
      v = atoi(argv[2]); 
   }  
   else
     {
       printf ("Sintaxis: <ejecutable> <total number of elements> <elementos del vector por thread>\n");
       exit(0);
     }

   if(n%v != 0){
      printf("El número de componentes del vector por thread no es divisor del total de elementos del mismo\n");
      exit(0);
   }

   srandom(12345);

   // Define vectors at host
   vector A;
   vector B;
   vector C;
   vector P;

   vector cA;
   vector cB;
   vector cC;
   vector cP;

   sdkCreateTimer(&hTimer);
   sdkResetTimer(&hTimer);
   sdkStartTimer(&hTimer);

   // Load values into A
   A = (int *) malloc(n*sizeof(int));
   cudaMalloc((void**)&cA,n*sizeof(int));
   LoadStartValuesIntoVectorRand(A,n);
   cudaMemcpy(cA, A, n*sizeof(int), cudaMemcpyHostToDevice);
   //printf("\nPrinting Vector A  %d\n",n);
   //PrintVector(A,n);

   // Load values 
   B = (int *) malloc(n*sizeof(int));
   cudaMalloc((void**)&cB,n*sizeof(int));
   LoadStartValuesIntoVectorRand(B,n);
   cudaMemcpy(cB, B, n*sizeof(int), cudaMemcpyHostToDevice);
   //printf("\nPrinting Vector B  %d\n",n);
   //PrintVector(B,n);

   C = (int *) malloc(n*sizeof(int));
   cudaMalloc(&cC,n*sizeof(int));

    // Load values 
   P = (int *) malloc(n*sizeof(int));
   cudaMalloc((void**)&cP,n*sizeof(int));
   LoadP(P,n);
   cudaMemcpy(cP, P, n*sizeof(int), cudaMemcpyHostToDevice);

   sdkCreateTimer(&kTimer);
   sdkResetTimer(&kTimer);
   sdkStartTimer(&kTimer);

   //printf("Llega\n");

   // execute the subprogram
   SumVectorCuda<<<n/(1024*v),1024>>>(cA,cB,cC,cP,n,v);

   cudaDeviceSynchronize();

   sdkStopTimer(&kTimer);

   //printf("Llega 2\n");

   //Copiar de dispositivo a host
   cudaMemcpy(C, cC, n*sizeof(int), cudaMemcpyDeviceToHost);

   //printf("Copia\n");

   //printf("\nPrinting vector C  %d\n",n);
   //PrintVector(C,n);
   

   // Free vectors
   free(A);
   free(B);
   free(C);
   free(P);

   cudaFree(cA);
   cudaFree(cB);
   cudaFree(cC);
   cudaFree(cP);

   sdkStopTimer(&hTimer);

   //cambiar timers a los de cuda

   timerValue = sdkGetTimerValue(&kTimer);
   timerValue = timerValue / 1000;
   sdkDeleteTimer(&kTimer);
   printf("Tiempo kernel: %f s", timerValue);
   ops = n/timerValue;
   printf("    %f GFLOPS\n",(ops)/1000000000);
   timerValue = sdkGetTimerValue(&hTimer);
   timerValue = timerValue / 1000;
   sdkDeleteTimer(&hTimer);
   printf("Tiempo total: %f s", timerValue);
   ops = n/timerValue;
   printf("    %f GFLOPS \n",(ops)/1000000000);

   return 0;
}
