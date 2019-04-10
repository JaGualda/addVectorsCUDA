// Suma de vectores secuencial
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


typedef int *vector;


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

// Suma vectores C = A + B
void SumVectorSec(vector A, vector B, vector C, unsigned int n)
{
   unsigned int k;
   
   for (k = 0; k <n ; k++)
      C[k] = A[k] + B[k];
}


// ------------------------
// MAIN function
// ------------------------
int main(int argc, char **argv)
{
   struct timeval start, stop;
   float timet;
   unsigned int n;

   if (argc == 2)
      n = atoi(argv[1]);
   else
     {
       printf ("Sintaxis: <ejecutable> <total number of elements>\n");
       exit(0);
     }

   srandom(12345);

   // Define vectors at host
   vector A;
   vector B;
   vector C;

   gettimeofday(&start,0);

   // Load values into A
   A = (int *)malloc(n*sizeof(int));
   LoadStartValuesIntoVectorRand(A,n);
   //printf("\nPrinting Vector A  %d\n",n);
   //PrintVector(A,n);

   // Load values 
   B = (int *)malloc(n*sizeof(int));
   LoadStartValuesIntoVectorRand(B,n);
   //printf("\nPrinting Vector B  %d\n",n);
   //PrintVector(B,n);

   C = (int *)malloc(n*sizeof(int));


   // execute the subprogram
   SumVectorSec(A,B,C,n);

   //printf("\nPrinting vector C  %d\n",n);
   //PrintVector(C,n);

   // Free vectors
   free(A);
   free(B);
   free(C);

   gettimeofday(&stop,0);
   timet = (stop.tv_sec + stop.tv_usec * 1e-6)-(start.tv_sec + start.tv_usec * 1e-6);
   printf("Time = %f s\n",timet);

   return 0;
}
