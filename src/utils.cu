#include <ccut/utils.cuh>
#include "nvidia_utilities.cuh"

#include <utilities.h>

#include <log.h>
#include <Assertion.h>

namespace {
//number of blocks in grid if have to use 2D grid
const size_t BLOCK_IN_X_DIV = 4; //I hope it will be about 16000 
}

namespace Cuda
{
	void Init(){
		//TODO fix
		checkCudaErrors(cudaSetDevice(0));
	};


	size_t RoundToBunch(size_t x, size_t bunch) 
	{
		return x%bunch == 0 ? x : (x/bunch+1)*bunch; 
	} 

	
	void Sync(){
		Assert(cudaSuccess == cudaDeviceSynchronize());
	}
}

dim3 CudaParams::m_maxBlocks = 1;
dim3 CudaParams::m_maxThreads = 1;


CudaParams::CudaParams(int dataSize, 
						size_t THREADS,
						bool yesIknowWhatImDoing)
{
	static bool init = CudaParams::init();

	if(dataSize < 0)
	{
		LogError("Wrong data size " << dataSize);
	}

	//For release - when Asserts disappear
	nThreads = 1;
	nBlocks = 1;

	//-------
	//THREADS
	//-------
	if(THREADS > m_maxThreads.x)
	{
		LogCritical("Wrong Threads nb");
		Assert(0);
	}
	nThreads = THREADS;

	//------
	//BLOCKS
	//------
	size_t roundedN = Cuda::RoundToBunch(dataSize,THREADS); 
	size_t _nBlocks = roundedN / nThreads.x;

	//1D;
	if(_nBlocks <=m_maxBlocks.x)
	{
		nBlocks.x = _nBlocks;
		return;
	}
	
	if (!yesIknowWhatImDoing)
	{
		LogCritical("ERROR: Problem with data size detected");
		Assert(0);
	}
	
	//2D
	nBlocks.x = m_maxBlocks.x / BLOCK_IN_X_DIV;
	nBlocks.y = _nBlocks % nBlocks.x == 0 ? _nBlocks / nBlocks.x : (_nBlocks / nBlocks.x)+1;  

	if (nBlocks.y <= m_maxBlocks.y)
	{
		return;
	}
	
	//3D
	//TODO implement me
	LogCritical("ERROR: Problem with data size detected");
	Assert(0);
}


bool CudaParams::init()
{
	cudaDeviceProp deviceProp;
	int dev=0;
	cudaGetDeviceProperties(&deviceProp, dev);

	m_maxBlocks = dim3(deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	m_maxThreads = dim3(deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

	return true;
}
