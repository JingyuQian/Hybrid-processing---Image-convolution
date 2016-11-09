#!/usr/bin/python
# -*- coding: utf-8 -*-
#EECS 4750
#JQ2250 Jingyu Qian
#HW2 CUDA version
#=====================================================================#
#Part1. Matrix transpose. Input a matrix and calculate its transpose.
#Part2-1. Naive matrix multiplication A*A.T
#Part2-2. Two kinds of matrix multiplication optimizations
#Part2-3. Plotting part
#=====================================================================#
#Import necessary modules
import pycuda
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import time
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
#=====================================================================#
#Define kernel functions.
#mat_trans: matrix transpose kernel
#mat_mul: naive matrix multiplication kernel
#m_mul1: matrix multiplication optimized kernel 1. Put one row of matrix
#       A into local memory
#m_mul2: matrix multiplication optimized kernel 2. Put one row of matrix
#       A into local memory and one column of matrix B into shared me-
#       mory
kernel_code_template="""
__global__ void mat_trans(float *a, float *b, unsigned int DIM)
{
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    b[tx*DIM+ty]=a[ty*DIM+tx];
}

__global__ void mat_mul(float *a, float *b, float *c, int m, int n)
{
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    float cvalue=0;
    for (int k=0;k<n;++k)
    {
        float Aelement=a[ty*n+k];
        float Belement=b[k*m+tx];
        cvalue+=Aelement*Belement;
    }
    c[ty*m+tx]=cvalue;
}

__global__ void m_mul1(int M, int N, float*A, float *B, float *C)
{
    float Awk_row[%(LENGTH)s];
    int row=threadIdx.y;
    
    for(int k=0;k<N;k++)
        Awk_row[k]=A[row*N+k];
    float tmp;
    for(int j=0;j<M;j++)
    {    
        tmp=0.0f;
        for(int k=0;k<N;k++)
        {
            tmp=tmp+Awk_row[k]*B[k*M+j];
        }
        C[row*M+j]=tmp;
    }
}

__global__ void m_mul2(unsigned int M, unsigned int N, float *A, float *B, float *C)
{
    __shared__ float ds_A[%(TILE_WIDTH)s][%(TILE_WIDTH)s];
    __shared__ float ds_B[%(TILE_WIDTH)s][%(TILE_WIDTH)s];

    int bx=blockIdx.x;
    int by=blockIdx.y;
    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int Row=by*blockDim.y+ty;
    int Col=bx*blockDim.x+tx;
    float tmp=0.0f;
    for(int t=0;t<N/%(TILE_WIDTH)s;t++)
    {
    ds_A[ty][tx]=A[Row*N+t*%(TILE_WIDTH)s+tx];
    ds_B[ty][tx]=B[(t*%(TILE_WIDTH)s+ty)*M+Col];
    __syncthreads();
    for(int i=0;i<%(TILE_WIDTH)s;i++)
        tmp+=ds_A[ty][i]*ds_B[i][tx];
    __syncthreads();
    }
    C[Row*M+Col]=0.0f;
    C[Row*M+Col]=tmp;
}
"""
#=====================================================================#

#Since some constants need to be passed after they're given values in 
#the host code, we can't compile the module now until values are passed

#=====================================================================#
#Initialize a sequence of matrix dimension
#The base dim I use is M:N=2:4. Each time use one DIM element to multi-
#ply base M and N to generate different matrix dimensions.
DIM=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#Initialize an array times_trans to record transpose time cost
#row 0: time taken by CPU transpose
#row 1: time taken by GPU transpose using mattrans kernel
times_trans=np.zeros((2,len(DIM)),dtype=np.float32)
#Initialize an array times_mult to record multiplication time cost
#row 0: time taken by CPU multiplication
#row 1: time taken by GPU naive multiplication
#row 2: time taken by GPU optimization 1 mmul1 kernel
#row 3: time taken by GPU optimization 2 mmul2 kernel
times_mult=np.zeros((4,len(DIM)),dtype=np.float32)
#=====================================================================#
for index,i in enumerate(DIM):
#At this point constants will be passed into kernels
#Define TILE_WIDTH which is passed into the kernel
    TILE_WIDTH=i
    kernel_code=kernel_code_template % {
        'LENGTH':4*i,
        'TILE_WIDTH':TILE_WIDTH
    }
#build the program and extract kernel functions
    mod=compiler.SourceModule(kernel_code)
    mattrans=mod.get_function('mat_trans')
    matmul=mod.get_function('mat_mul')
    mmul1=mod.get_function('m_mul1')
    mmul2=mod.get_function('m_mul2')
#=======================Part1 CPU transpose===========================#
#Initialize input matrix a_cpu with random numbers. A is square
    a_cpu=np.random.randint(0,10,size=(i,i)).astype(np.float32)
    a_transpose=np.empty_like(a_cpu).astype(np.float32)
    start=time.time()
    for j in range(i):
        for k in range(i):
            a_transpose[j,k]=a_cpu[k,j] 
    times_trans[0][index]=time.time()-start
#=======================Part1 GPU transpose===========================#
#transfer data into device
    a_gpu=gpuarray.to_gpu(a_cpu)
    b_cpu=np.empty_like(a_cpu)
    b_gpu=gpuarray.to_gpu(b_cpu)
    start=time.time()
    mattrans(a_gpu,b_gpu,np.uint32(i),block=(i,i,1))
    times_trans[1][index]=time.time()-start
#========================Part1 validation=============================#    
    b_cpu=b_gpu.get()
    print i,'*',i,' Transpose comparison:',np.allclose(a_transpose,b_cpu)
#================ Part2-1 Naive Multiplication========================#
#Re-initialize a_cpu to be a non-square matrix with size 2i*4i
#b_cpu is the transpose of a_host with size 4i*2i
    a_cpu=np.random.randint(0,10,size=(2*i,4*i)).astype(np.float32)
    b_cpu=np.zeros((4*i,2*i),dtype=np.float32)
    for j in range(4*i):
        for k in range(2*i):
            b_cpu[j,k]=a_cpu[k,j]
#====================Part2-1 CPU Multiplication=======================#
#The result can be used for comparison in the following optimizations.
    a_prod=np.zeros((2*i,2*i),dtype=np.float32)
    start=time.time()
    for j in range(2*i):
        for k in range(2*i):
            for m in range(4*i):
                a_prod[j,k]=a_prod[j,k]+a_cpu[j,m]*b_cpu[m,k]
    times_mult[0][index]=time.time()-start
#====================Part2-1 GPU Multiplication=======================#
#Transfer data to device
    a_gpu=gpuarray.to_gpu(a_cpu)
    b_gpu=gpuarray.to_gpu(b_cpu)
    c_gpu=gpuarray.empty((2*i,2*i),np.float32)
    start=time.time()
    matmul(a_gpu,b_gpu,c_gpu,np.uint32(2*i),np.uint32(4*i),block=(2*i,2*i,1))
    times_mult[1][index]=time.time()-start
#======================Part2-1 GPU Validation=========================#
    c_cpu=c_gpu.get()
    print 2*i,'*',4*i,' Multiplication comparison:',np.allclose(a_prod,c_cpu)
#============================Part2-2==================================#

#======================Part2-2 Optimization1==========================#
#a_gpu and b_gpu and c_gpu can be used directly since they don't change
    knl_grid=(1,1,1)
    knl_block=(1,2*i,1)
    start=time.time()
    mmul1(np.uint32(2*i),np.uint32(4*i),a_gpu,b_gpu,c_gpu,grid=knl_grid,block=knl_block)
    times_mult[2][index]=time.time()-start
#==================Part2-2 Optimization1 Validation===================#
    c_cpu=c_gpu.get()
    print 2*i,'*',4*i,' Optimized Multiplication 1 comparison:',np.allclose(a_prod,c_cpu)
#======================Part2-2 Optimization2==========================#
    start=time.time()
    knl_grid=(2*i/TILE_WIDTH,2*i/TILE_WIDTH,1)
    knl_block=(TILE_WIDTH,TILE_WIDTH,1)
    mmul2(np.uint32(2*i),np.uint32(4*i),a_gpu,b_gpu,c_gpu,grid=knl_grid,block=knl_block)
    times_mult[3][index]=time.time()-start
#==================Part2-2 Optimization1 Validation===================#
    c_cpu=c_gpu.get()
    print 2*i,'*',4*i,' Optimized Multiplication 2 comparison:',np.allclose(a_prod,c_cpu)
#=====================Part2-3 Plotting Results========================#
num_elements=[i*i*8 for i in DIM]
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.plot(num_elements,times_trans[0,:],num_elements,times_trans[1,:])
plt.xlabel('Number of elements of input matrix')
plt.ylabel('Computation time')
plt.legend(['CPU time','GPU time'])
plt.title('Matrix Transpose(CUDA)')
plt.gca().set_xlim((min(num_elements),max(num_elements)))
plt.subplot(2,1,2)
plt.plot(num_elements,times_mult[0,:],num_elements,times_mult[1,:],num_elements,times_mult[2,:],num_elements,times_mult[3,:])
plt.xlabel('Number of elements of input matrix')
plt.ylabel('Computation time')
plt.legend(['CPU time','GPU time(naive)','GPU time(opt1)','GPU time(opt2)'])
plt.title('Matrix Multiplication(CUDA)')
plt.gca().set_xlim((min(num_elements),max(num_elements)))
plt.savefig('CUDA_results.png')
plt.close()
plt.figure(figsize=(12,12))
plt.plot(num_elements,times_mult[1,:],num_elements,times_mult[2,:],num_elements,times_mult[3,:])
plt.xlabel('Number of elements of input matrix')
plt.ylabel('Computation time')
plt.legend(['GPU time(naive)','GPU time(opt1)','GPU time(opt2)'])
plt.title('Matrix Multiplication Magnified(CUDA)')
plt.gca().set_xlim((min(num_elements),max(num_elements)))
plt.savefig('CUDA_results_magnified.png')

