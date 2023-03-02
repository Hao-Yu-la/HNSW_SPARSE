#include<iostream>
#include<cstdlib>
#include<ctime>
#include<stdio.h>
typedef float dtype;
typedef int itype;
using namespace std;

void csr_to_matrix(dtype *value, itype *colindex, itype *rowptr, int n, int m, dtype** & M){
    M=new dtype*[n];
    for(int i=0;i<n;i++)
        M[i]=new dtype[m];
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            M[i][j]=0;
    for(int i=0;i<n;i++)
        for(int j=rowptr[i];j<rowptr[i+1];j++)
            M[i][colindex[j]]=value[j];
    return;
}

void spmv(dtype *value,itype *rowptr,itype *colindex,int n,dtype *x,dtype *y){
    //calculate the matrix-vector multiply where matrix is stored in the form of CSR
    for(int i=0;i<n;i++){
        dtype y0=0;
        for(int j=rowptr[i];j<rowptr[i+1];j++)
            y0+=value[j]*x[colindex[j]];
        y[i]=y0;
    }
    return;
}

int matrix_to_csr(int n,int m,dtype **M,dtype* &value,itype* & rowptr,itype* & colindex){
   int i,j;
   int a=0;
   for(i=0;i<n;i++)
      for(j=0;j<m;j++)
          if(M[i][j]!=0)
              a++;
   value=new dtype[a];
   colindex=new int[a];
   rowptr=new int[n+1];
   int k=0;
   int l=0;
   for(i=0;i<n;i++)
      for(j=0;j<m;j++){
          if(j==0)
              rowptr[l++]=k;
          if(M[i][j]!=0){
              value[k]=M[i][j];
              colindex[k]=j;
              k++;}
   }
   rowptr[l]=a;
   return a;
}

void matrix_multiply_vector(dtype **m,int n,dtype *x,dtype *y){
   for(int i=0;i<n;i++)
   {
       dtype y0=0;
       for(int j=0;j<n;j++)
           y0+=m[i][j]*x[j];
       y[i]=y0;
   }
   return;
}
