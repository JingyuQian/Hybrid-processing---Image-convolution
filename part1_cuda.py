#!/usr/bin/python
# -*- coding: utf-8 -*-
#EECS 4750
#JQ2250 Jingyu Qian
#HW3 CUDA version
#=====================================================================#
#Part1. Implement a 2D convolution kernel and compare the result with
#scipy.signal.convolve2d function.
#Part2. Keep kernel constant and test the speed with different matrix
#Part3. Keep matrix constant and test the speed with different kernel
#=====================================================================#
#Import necessary modules
from PIL import Image
import pycuda
import numpy as np
import scipy as sp
import scipy.signal
from scipy.signal import convolve2d as conv2d
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import time
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
#=====================================================================#
filters = {
                'identity':np.array([ [0.,0.,0.],[0.,1.,0.],[0.,0.,0.]  ]).astype(np.int32),
                'sharpen':np.array([[0., -1. , 0.], [-1., 5., -1.], [0., -1., 0]]).astype(np.int32),
                'blur':np.array([[1., 1. , 1.], [1., 1., 1.], [1., 1., 1.]]).astype(np.int32),
                'edge_det':np.array([[0., 1. , 0.], [1., -4., 1.], [0., 1., 0]]).astype(np.int32),
                'emboss':np.array([[2., 1. , 0.], [1., 1., -1.], [0., -1., -2]]).astype(np.int32),
                'sob_x':np.array([[-1., 0. , 1.], [-2., 0., 2.], [-1., 0., 1]]).astype(np.int32),
                'sob_y':np.array([[-1., -2. , -1.], [0., 0., 0.], [1., 2., 1.]]).astype(np.int32),
                'smooth_5x5':np.array([[0., 1., 2. , 1., 0.], [1., 4., 8., 4., 1.],[2.,8.,16.,8.,2.],[1.,4.,8.,4.,1.], [0.,1., 2., 1.,0.]]).astype(np.int32)
          }
#=====================================================================#
kernel_code_template="""
#define O_TILE_WIDTH %(O_TILE_WIDTH)s
#define BLOCK_WIDTH %(BLOCK_WIDTH)s
#define MASK_WIDTH %(MASK_WIDTH)s
__global__ void conv_2D(int *P,
                        int *N,
                        const int *__restrict__ M,
                        int height,
                        int width,
                        int m_size)
{
    __shared__ int Ns[BLOCK_WIDTH][BLOCK_WIDTH]; 
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int row_o=blockIdx.y*O_TILE_WIDTH+ty;
    int col_o=blockIdx.x*O_TILE_WIDTH+tx;
    int row_i=row_o-m_size/2;
    int col_i=col_o-m_size/2;
    if((row_i>=0)&&(row_i<height)&&(col_i>=0)&&(col_i<width)){
        Ns[ty][tx]=P[row_i*width+col_i];
    }
    else{
    Ns[ty][tx]=0;
    }
    __syncthreads();
    if(ty<O_TILE_WIDTH && tx<O_TILE_WIDTH){
        int output=0;
        for(int i=0;i<MASK_WIDTH;i++){
            for(int j=0;j<MASK_WIDTH;j++){
                output+=M[i*m_size+j]*Ns[i+ty][j+tx];
            }
        }
        if(row_o<height && col_o<width){
            N[row_o*width+col_o]=output;
        }
    }
}
"""
#=====================================================================#
def create_img(filename,cols,rows):
    size=(cols,rows)
    im=Image.open(filename).convert('L')
    im=im.resize(size)
    return np.array(im)
#=====================================================================#
#Part1-(i) Compare the performance with scipy.signal.convolve2d function
#The comparison uses a 3*3 mask and a 4*4 matrix.
#The mask applied is a symmetric mask so doesn't need to be reversed.
#In the other questions the mask will have to be reversed.
mask=np.ones((3,3),dtype=np.int32)
mask[1,0]=0
mask[1,2]=0
#Passing constants
kernel_code=kernel_code_template % {
    'MASK_WIDTH':3,
    'O_TILE_WIDTH':2,
    'BLOCK_WIDTH':4
}
#Compiling functions
mod=compiler.SourceModule(kernel_code)
conv=mod.get_function('conv_2D')
#Transporting data into GPU & prepare host side output
i_cpu=np.random.randint(0,10,size=(4,4)).astype(np.int32)
o_cpu=np.empty((4,4)).astype(np.int32)
i_gpu=gpuarray.to_gpu(i_cpu)
mask_gpu=gpuarray.to_gpu(mask)
o_gpu=gpuarray.empty((4,4),np.int32)
#Launch kernel
conv(i_gpu,o_gpu,mask_gpu,np.uint32(4),np.uint32(4),np.uint32(3),grid=(2,2,1),block=(4,4,1))
#Validation
o_cpu=o_gpu.get()
o_reference=conv2d(i_cpu,mask,mode='same')
print('===Part1-(i)===')
print('===Input Matrix===')
print(i_cpu)
print('===Input Mask===')
print(mask)
print('===GPU Output===')
print(o_cpu)
print('===CPU Output===')
print(o_reference)
print 'Validation result: ',np.allclose(o_cpu,o_reference)
#=====================================================================#
#Part1-(ii) Test with different matrix size, keeping kernel constant.
#Specify matrix size to be used
print('===Part1-(ii)===')
size=np.array([[20,40,60,80,100,120,140,160,180,200,300,400,500,600],
               [10,20,30,40,50,60,70,80,90,100,200,300,400,500]])
times_consA=np.zeros((1,14),dtype=np.float32)
#Kernel is 5*5 in size. This time a reversed kernel should be used
#because it's not symmetric any more. Reverse it outside GPU.
mask=np.random.randint(1,10,size=(5,5)).astype(np.int32)
mask_reversed=np.zeros((5,5),dtype=np.int32)
for i in range(0,5):
    for j in range(0,5):
	mask_reversed[i,j]=mask[4-i,4-j]
#Specify constants and recompile the kernel
mask_width=5
o_tile_width=5
block_width=9
kernel_code=kernel_code_template % {
    'MASK_WIDTH':mask_width,
    'O_TILE_WIDTH':o_tile_width,
    'BLOCK_WIDTH':block_width
}
mod=compiler.SourceModule(kernel_code)
conv=mod.get_function('conv_2D')
#Test with different matrix size
for index in range(0,14):
    width=size[0,index]
    height=size[1,index]
#Setting GPU data transfer & host side output
    i_cpu=np.random.randint(1,10,size=(height,width)).astype(np.int32)
    o_cpu=np.empty_like(i_cpu)
    i_gpu=gpuarray.to_gpu(i_cpu)
    o_gpu=gpuarray.empty((height,width),np.int32)
    mask_gpu=gpuarray.to_gpu(mask_reversed)
    blockdim=(block_width,block_width,1)
    griddim=((width-1)//o_tile_width+1,(height-1)//o_tile_width+1,1)
#Launch the kernel. Timer start
    start=time.time()
    conv(i_gpu,o_gpu,mask_gpu,np.uint32(height),np.uint32(width),np.uint32(mask_width),grid=griddim,block=blockdim) 
    times_consA[0,index]=time.time()-start
#Timer stop
#Validation
    o_cpu=o_gpu.get()
    o_reference=conv2d(i_cpu,mask,mode='same')
    print 'Matrix size %d * %d'%(height,width),np.allclose(o_cpu,o_reference)
print('Time Taken')
print(times_consA)
#=====================================================================#
#Part1-(iii) Test with different kernel size,keeping matrix constant.
print('===Part iii===')
#Choose a 200*300 matrix to test
i_cpu=np.random.randint(1,10,size=(200,300)).astype(np.int32)
#Specify kernel sizes to be tested
mask_size=np.array([3,5,7,9,11,13,15,17,19,21,23,25])
times_consK=np.zeros((1,12),dtype=np.float32)
width=300
height=200
for index,i in enumerate(mask_size):
#Every iteration changes constants in the kernel
    mask_width=i
    block_width=o_tile_width+mask_width-1
    kernel_code=kernel_code_template % {
        'MASK_WIDTH':mask_width,
        'O_TILE_WIDTH':o_tile_width,
        'BLOCK_WIDTH':block_width
    }
    mod=compiler.SourceModule(kernel_code)
    conv=mod.get_function('conv_2D')
    mask=np.random.randint(1,10,size=(i,i)).astype(np.int32)
    mask_reversed=np.zeros((i,i),dtype=np.int32)
    for j in range(0,i):
        for k in range(0,i):
            mask_reversed[j,k]=mask[i-j-1,i-k-1]
#Transfer data to GPU & prepare host side output
    i_gpu=gpuarray.to_gpu(i_cpu)
    o_cpu=np.empty_like(i_cpu)
    o_gpu=gpuarray.empty((height,width),np.int32)
    mask_gpu=gpuarray.to_gpu(mask_reversed)
    blockdim=(block_width,block_width,1)
    griddim=((width-1)//o_tile_width+1,(height-1)//o_tile_width+1,1)
#Launch the kernel. Timer start
    start=time.time()
    conv(i_gpu,o_gpu,mask_gpu,np.uint32(height),np.uint32(width),np.uint32(mask_width),grid=griddim,block=blockdim)
    times_consK[0,index]=time.time()-start
#Timer stop
    o_cpu=o_gpu.get()
    o_reference=conv2d(i_cpu,mask,mode='same')
    print 'Mask size: %d * %d'%(mask_width,mask_width),np.allclose(o_cpu,o_reference)
print('Time Taken')
print(times_consK)
#=====================================================================#
#Part1 Plotting
plt.figure(figsize=(12,12))
plt.plot(mask_size,times_consK[0,:])
plt.xlabel('Mask Size')
plt.ylabel('Computation Time')
plt.legend(['GPU time curve'])
plt.title('Convolution Time VS. Different Kernel Sizes (Range 1)')
plt.xlim((2,26))
plt.savefig('CUDA_Diff_Kernel_size.png')
plt.close()
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.plot(size[0,0:10],times_consA[0,0:10])
plt.xlabel('Matrix Size')
plt.ylabel('Computation Time')
plt.legend(['GPU time curve'])
plt.title('Convolution Time VS. Different Matrix Sizes(Range 1)')
plt.subplot(2,1,2)
plt.plot(size[0,10:14],times_consA[0,10:14])
plt.xlabel('Matrix Size')
plt.ylabel('Computation Time')
plt.legend(['GPU time curve'])
plt.title('Convolution Time VS. Different Matrix Sizes(Range 2)')
plt.savefig('CUDA_Diff_Mat_size.png')
#=====================================================================#
#Part2
width=612
height=380
image=create_img('../thrones-002-2.jpg',width,height)
image=image.astype(np.int32)

for index in filters:
    name=index+'_original_GPU_CPU(CUDA).jpg'
    filter_chosen=filters[index]
    mask_width=filter_chosen.shape[0]
    o_tile_width=2
    block_width=o_tile_width+mask_width-1
    blockdim=(block_width,block_width,1)
    griddim=((width-1)//o_tile_width+1,(height-1)//o_tile_width+1,1)
    kernel_code=kernel_code_template % {
        'MASK_WIDTH':mask_width,
        'O_TILE_WIDTH':o_tile_width,
        'BLOCK_WIDTH':block_width
    }
    mod=compiler.SourceModule(kernel_code)
    conv=mod.get_function('conv_2D')
    filter_reversed=np.zeros((mask_width,mask_width),dtype=np.int32)
    for j in range(mask_width):
        for k in range(mask_width):
            filter_reversed[j,k]=filter_chosen[mask_width-j-1,mask_width-k-1]
    i_gpu=gpuarray.to_gpu(image)
    o_cpu=np.zeros((height,width),dtype=np.int32)
    o_gpu=gpuarray.to_gpu(o_cpu)
    mask_gpu=gpuarray.to_gpu(filter_reversed)
    conv(i_gpu,o_gpu,mask_gpu,np.uint32(height),np.uint32(width),np.uint32(mask_width),grid=griddim,block=blockdim)
    o_cpu=o_gpu.get()
    o_reference=conv2d(image,filter_chosen,mode='same')
    if(index=='smooth_5x5' or index=='blur'):
        o_reference=(o_reference-np.min(o_reference))*255/(np.max(o_reference)-np.min(o_reference))
        o_cpu=(o_cpu-np.min(o_cpu))*255/(np.max(o_cpu)-np.min(o_cpu))
    output=np.concatenate((image,o_cpu,o_reference),axis=1)
    new_image=Image.fromarray(output)
    new_image.convert('RGB').save(name)
