
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <iostream>

#include <stdio.h>

#define OpeningKernelSize 4
#define EroziuneKernelSize 4

#define TILE_WIDTH 16

__device__ int Gx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
__device__ int Gy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };



void addWithCuda(int *a, int *b, int rows, int cols);

__global__ void DilatareKernel(int *a,  int *b, int rows,  int cols)
{
	long Row = blockIdx.y*blockDim.y + threadIdx.y;
	long Col = blockIdx.x*blockDim.x + threadIdx.x;
	int c=0;
	if (blockIdx.x == 0 )
	{
		b[Col*cols + Row] = 0;
	}
	else
	{
		for (int i = -OpeningKernelSize; i <= OpeningKernelSize; i++)
		{
			for (int j = -OpeningKernelSize; j <= OpeningKernelSize; j++)
			{
				c += a[(Col + i)*cols + (Row + j)];
			}
		}
		if (c > 0)
		{
			b[Col*cols + Row] = 1;
		}
		else
		{
			b[Col*cols + Row] = 0;
		}
	}
}

__global__ void Skeletare1(int *a, int *b, int rows, int cols, bool* change)
{
	long Row = blockIdx.y*blockDim.y + threadIdx.y;
	long Col = blockIdx.x*blockDim.x + threadIdx.x;
	int c = 1;
	if (blockIdx.x == 0)
	{
		b[Col*cols + Row] = 0;
	}
	else
	{
		
		int tranz = 0;

		tranz += ((a[(Col - 1)*cols + Row] + a[(Col - 1)*cols + Row +1])%2);
		tranz += ((a[(Col - 1)*cols + Row+1] + a[(Col)*cols + Row + 1]) % 2);
		tranz += ((a[(Col )*cols + Row + 1] + a[(Col + 1)*cols + Row + 1]) % 2);
		tranz += ((a[(Col +1)*cols + Row + 1] + a[(Col + 1)*cols + Row ]) % 2);
		tranz += ((a[(Col + 1)*cols + Row ] + a[(Col + 1)*cols + Row - 1 ]) % 2);
		tranz += ((a[(Col + 1)*cols + Row - 1] + a[(Col )*cols + Row - 1]) % 2);
		tranz += ((a[(Col)*cols + Row - 1] + a[(Col -1)*cols + Row - 1]) % 2);
		tranz += ((a[(Col - 1)*cols + Row - 1] + a[(Col - 1)*cols + Row ]) % 2);

		int vecini = a[(Col - 1)*cols + Row] + a[(Col - 1)*cols + Row + 1] + a[(Col)*cols + Row + 1] + a[(Col + 1)*cols + Row + 1] + a[(Col + 1)*cols + Row] + a[(Col + 1)*cols + Row - 1] + a[(Col)*cols + Row - 1] + a[(Col - 1)*cols + Row - 1];

		if (a[(Col)*cols + Row] == 1 && vecini > 1 && vecini<7 && tranz == 2 && (a[(Col - 1)*cols + Row] == 0 || a[(Col)*cols + Row + 1] == 0 || a[(Col + 1)*cols + Row] == 0) &&
			(a[(Col)*cols + Row +1] == 0 || a[(Col+1)*cols + Row ] == 0 || a[(Col)*cols + Row-1] == 0))
		{
			b[Col*cols + Row] = 0;
			//change[0] = true;
		}
		else
		{
			b[Col*cols + Row] = 1;
		}
	}

}

__global__ void Skeletare2(int *a, int *b, int rows, int cols, bool* change)
{
	long Row = blockIdx.y*blockDim.y + threadIdx.y;
	long Col = blockIdx.x*blockDim.x + threadIdx.x;
	int c = 1;
	if (blockIdx.x == 0)
	{
		b[Col*cols + Row] = 0;
	}
	else
	{

		int tranz = 0;

		tranz += ((a[(Col - 1)*cols + Row] + a[(Col - 1)*cols + Row + 1]) % 2);
		tranz += ((a[(Col - 1)*cols + Row + 1] + a[(Col)*cols + Row + 1]) % 2);
		tranz += ((a[(Col)*cols + Row + 1] + a[(Col + 1)*cols + Row + 1]) % 2);
		tranz += ((a[(Col + 1)*cols + Row + 1] + a[(Col + 1)*cols + Row]) % 2);
		tranz += ((a[(Col + 1)*cols + Row] + a[(Col + 1)*cols + Row - 1]) % 2);
		tranz += ((a[(Col + 1)*cols + Row - 1] + a[(Col)*cols + Row - 1]) % 2);
		tranz += ((a[(Col)*cols + Row - 1] + a[(Col - 1)*cols + Row - 1]) % 2);
		tranz += ((a[(Col - 1)*cols + Row - 1] + a[(Col - 1)*cols + Row]) % 2);

		int vecini = a[(Col - 1)*cols + Row] + a[(Col - 1)*cols + Row + 1] + a[(Col)*cols + Row + 1] + a[(Col + 1)*cols + Row + 1] + a[(Col + 1)*cols + Row] + a[(Col + 1)*cols + Row - 1] + a[(Col)*cols + Row - 1] + a[(Col - 1)*cols + Row - 1];

		if (a[(Col)*cols + Row] == 1 && vecini > 1 && vecini<7 && tranz == 2 && (a[(Col - 1)*cols + Row] == 0 || a[(Col)*cols + Row + 1] == 0 || a[(Col )*cols + Row-1] == 0) &&
			(a[(Col - 1)*cols + Row] == 0 || a[(Col+1)*cols + Row ] == 0 || a[(Col )*cols + Row-1] == 0))
		{
			b[Col*cols + Row] = 0;
			//change[0] = true;
		}
		else
		{
			b[Col*cols + Row] = 1;
		}
	}

}

__global__ void EroziuneKernel(int *a, int *b, int rows, int cols)
{
	long Row = blockIdx.y*blockDim.y + threadIdx.y;
	long Col = blockIdx.x*blockDim.x + threadIdx.x;
	int c = 1;
	if (blockIdx.x == 0)
	{
		b[Col*cols + Row] = 0;
	}
	else
	{
		for (int i = -OpeningKernelSize; i <= EroziuneKernelSize; i++)
		{
			for (int j = -OpeningKernelSize; j <= EroziuneKernelSize; j++)
			{
				c *= a[(Col + i)*cols + (Row + j)];
			}
		}
		if (c > 0)
		{
			b[Col*cols + Row] = 1;
		}
		else
		{
			b[Col*cols + Row] = 0;
		}
	}
}

__global__ void thinningIterationStep1(int* im, int* marker,int rows, int cols )
{
	long Row = blockIdx.y*blockDim.y + threadIdx.y;
	long Col = blockIdx.x*blockDim.x + threadIdx.x;

	//long p2 = im[(Col - 1)*cols + Row];
	//int p3 = im[(Col - 1)*cols + Row+1];
	//int p4 = im[(Col)*cols + Row + 1];//im.at<int>(i, j + 1);
	//int p5 = im[(Col + 1)*cols + Row+1];//im.at<int>(i + 1, j + 1);
	//int p6 = im[(Col + 1)*cols + Row];//im.at<int>(i + 1, j);
	//int p7 = im[(Col + 1)*cols + Row-1];//im.at<int>(i + 1, j - 1);
	//int p8 = im[(Col)*cols + Row-1];//im.at<int>(i, j - 1);
	//int p9 = im[(Col - 1)*cols + Row-1];// im.at<int>(i - 1, j - 1);

	int A = (im[(Col - 1)*cols + Row] == 0 && im[(Col - 1)*cols + Row+1] == 1) + (im[(Col - 1)*cols + Row+1] == 0 && im[(Col)*cols + Row + 1] == 1) +
		(im[(Col)*cols + Row + 1] == 0 &&  im[(Col + 1)*cols + Row+1] == 1) + ( im[(Col + 1)*cols + Row+1] == 0 && im[(Col + 1)*cols + Row] == 1) +
		(im[(Col + 1)*cols + Row] == 0 && im[(Col + 1)*cols + Row] == 1) + (im[(Col + 1)*cols + Row] == 0 && im[(Col)*cols + Row-1] == 1) +
		(im[(Col)*cols + Row-1] == 0 && im[(Col - 1)*cols + Row-1] == 1) + (im[(Col - 1)*cols + Row-1] == 0 && im[(Col - 1)*cols + Row] == 1);
	int B = im[(Col - 1)*cols + Row] + im[(Col - 1)*cols + Row+1] + im[(Col)*cols + Row + 1] +  im[(Col + 1)*cols + Row+1] + im[(Col + 1)*cols + Row] + im[(Col + 1)*cols + Row] + im[(Col)*cols + Row-1] + im[(Col - 1)*cols + Row-1];
	int m1 =  (im[(Col - 1)*cols + Row] * im[(Col)*cols + Row + 1] * im[(Col + 1)*cols + Row]) ;
	int m2 =  (im[(Col)*cols + Row + 1] * im[(Col + 1)*cols + Row] * im[(Col)*cols + Row-1]) ;

	if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
	{
		marker[Col*cols + Row] = 1;

	}
}

__global__ void thinningIterationStep2(int* im, int* marker, int rows, int cols)
{
	long Row = blockIdx.y*blockDim.y + threadIdx.y;
	long Col = blockIdx.x*blockDim.x + threadIdx.x;

	int A = (im[(Col - 1)*cols + Row] == 0 && im[(Col - 1)*cols + Row + 1] == 1) + (im[(Col - 1)*cols + Row + 1] == 0 && im[(Col)*cols + Row + 1] == 1) +
		(im[(Col)*cols + Row + 1] == 0 && im[(Col + 1)*cols + Row + 1] == 1) + (im[(Col + 1)*cols + Row + 1] == 0 && im[(Col + 1)*cols + Row] == 1) +
		(im[(Col + 1)*cols + Row] == 0 && im[(Col + 1)*cols + Row] == 1) + (im[(Col + 1)*cols + Row] == 0 && im[(Col)*cols + Row - 1] == 1) +
		(im[(Col)*cols + Row - 1] == 0 && im[(Col - 1)*cols + Row - 1] == 1) + (im[(Col - 1)*cols + Row - 1] == 0 && im[(Col - 1)*cols + Row] == 1);
	int B = im[(Col - 1)*cols + Row] + im[(Col - 1)*cols + Row + 1] + im[(Col)*cols + Row + 1] + im[(Col + 1)*cols + Row + 1] + im[(Col + 1)*cols + Row] + im[(Col + 1)*cols + Row] + im[(Col)*cols + Row - 1] + im[(Col - 1)*cols + Row - 1];

	int m1 = (im[(Col - 1)*cols + Row] * im[(Col)*cols + Row + 1] * im[(Col)*cols + Row-1]);
	int m2 = (im[(Col - 1)*cols + Row] * im[(Col + 1)*cols + Row] * im[(Col)*cols + Row-1]);

	if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
	{
		
		marker[Col*cols + Row] = 1;
	}
}

__global__ void applyMask(int*im, int*marker, int rows, int cols)
{
	long Row = blockIdx.y*blockDim.y + threadIdx.y;
	long Col = blockIdx.x*blockDim.x + threadIdx.x;

	im[Col*cols + Row] = im[Col*cols + Row] & ~marker[Col*cols + Row];
}

__global__ void SobelKernel(int *a, int *b, int rows, int cols)
{
	long Row = blockIdx.y*blockDim.y + threadIdx.y;
	long Col = blockIdx.x*blockDim.x + threadIdx.x;
	int CGx = 0, CGy = 0;
	int c = 0;
	

	if (blockIdx.x == 0)
	{
		b[Col*cols + Row] = 0;
	}
	else
	{
		for (int i = 0; i <3; i++)
		{
			for (int j = 0; j <3; j++)
			{
				CGx += a[(Col - 1 + i)*cols + (Row - 1 + j)] * Gx[i*3+j];
				CGy += a[(Col - 1 + i)*cols + (Row - 1 + j)] * Gy[i * 3 + j];
			}
		}
		c = sqrtf(CGx*CGx + CGy*CGy);
		if (c > 0)
		{
			b[Col*cols + Row] = 1;
		}
		else
		{
			b[Col*cols + Row] = 0;
		}
	}

}


void addWithCuda( int *a, int *b,  int rows,  int cols)
{
    int *dev_a = 0;
    int *dev_b = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, rows * cols*sizeof(int));
    if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaMalloc failed!");
        goto Error;
    }


	cudaStatus = cudaMalloc((void**)&dev_a, rows * cols* sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, rows*cols * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		fprintf(stdout,"cudaMemcpy failed!");
        goto Error;
    }


	dim3 DimGrid((rows - 1) / TILE_WIDTH + 1, (cols - 1) / TILE_WIDTH + 1, 1);
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	DilatareKernel <<<DimGrid, DimBlock>>>(dev_a, dev_b, rows, cols);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "DilatareKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
		
	cudaMemcpy(dev_a, dev_b, rows*cols * sizeof(int), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaMemcpy failed!");
		goto Error;
	}

	EroziuneKernel <<<DimGrid, DimBlock>>>(dev_a, dev_b, rows, cols);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "EroziuneKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, dev_b, rows*cols * sizeof(int), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout,"cudaMemcpy failed!");
		goto Error;
	}

	int nrPixelsPrev = 0, nrpixels = 0;

	for (int i = 0; i < rows*cols; i++)
	{
		if (a[i] != 0)
			nrpixels++;
	}


	//do
	//{
	//	nrPixelsPrev = nrpixels;
	//	thinningIterationStep1 <<<DimGrid, DimBlock>>>(dev_a, dev_b, rows, cols);
	//	cudaStatus = cudaGetLastError();
	//	if (cudaStatus != cudaSuccess) {
	//		fprintf(stdout, "EroziuneKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	//		goto Error;
	//	}
	//	applyMask <<<DimGrid, DimBlock>>>(dev_a, dev_b, rows, cols);
	//	cudaStatus = cudaGetLastError();
	//	if (cudaStatus != cudaSuccess) {
	//		fprintf(stdout, "EroziuneKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	//		goto Error;
	//	}
	//	thinningIterationStep2 <<<DimGrid, DimBlock>>>(dev_a, dev_b, rows, cols);
	//	cudaStatus = cudaGetLastError();
	//	if (cudaStatus != cudaSuccess) {
	//		fprintf(stdout, "EroziuneKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	//		goto Error;
	//	}
	//	applyMask <<<DimGrid, DimBlock>>>(dev_a, dev_b, rows, cols);
	//	cudaStatus = cudaGetLastError();
	//	if (cudaStatus != cudaSuccess) {
	//		fprintf(stdout, "EroziuneKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	//		goto Error;
	//	}
	//	cudaMemcpy(a, dev_a, rows*cols*sizeof(int), cudaMemcpyDeviceToHost);
	//	for (int i = 0; i < rows*cols; i++)
	//	{
	//		if (a[i] == 0)
	//			nrpixels++;
	//	}
	//	test++;
	//} while (test<100);

	//SobelKernel <<<DimGrid, DimBlock>>>(dev_a, dev_b, rows, cols);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
		fprintf(stdout,"SobelKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
 
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
       fprintf(stdout,"cudaDeviceSynchronize returned error code %d after launching DilatareKernel!\n", cudaStatus);
        goto Error;
    }

    
    cudaStatus = cudaMemcpy(b, dev_b, rows*cols * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
		fprintf(stdout,"cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    
	return;
}
