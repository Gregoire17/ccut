#ifndef _UTILITIES_CUDA_H_
#define _UTILITIES_CUDA_H_

#include <cufft.h>


namespace Cuda
{
	void Init();

	size_t RoundToBunch(size_t x, size_t bunch);

	void  Sync();
}



class CudaParams
{
public:
	explicit CudaParams(int dataSize, 
						size_t THREADS = 256,
						bool yesIknowWhatIamDoing = false);
	
	dim3 nThreads;
	dim3 nBlocks;

private:
	bool init();

	static dim3 m_maxThreads;
	static dim3 m_maxBlocks;
};

#endif
