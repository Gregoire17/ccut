#ifndef _UTILITIES_MEMORY_H_
#define _UTILITIES_MEMORY_H_

//shared_ptr
#include <boost/shared_ptr.hpp>

#include <ccut/memory_impl.cuh>

#include <complex_t.h>

//checkCudaError
#include <ccut/utils.cuh>


/**
* Typedefs
*/

/*
Int
*/
typedef MemoryDevice<int> IntD;
typedef boost::shared_ptr<IntD> IntDPtr;

/*
Floats
*/
typedef MemoryDevice<float> FloatD;
typedef MemoryHost<float>   FloatH;
typedef boost::shared_ptr<FloatD> FloatDPtr;
typedef boost::shared_ptr<FloatH> FloatHPtr;

/*
Complex
*/
typedef MemoryDevice<ComplexFloat> ComplexD;
typedef MemoryHost<ComplexFloat>   ComplexH;

typedef boost::shared_ptr<ComplexD> ComplexDPtr;
typedef boost::shared_ptr<ComplexH> ComplexHPtr;


/**
* REFERENCES
*/
/*
Host Pointers - without allocation
*/
typedef HostRef<ComplexFloat>		ComplexHRef;
typedef HostRef<float>				FloatHRef;

/*
Device Pointers - without allocation
*/
typedef DeviceRef<ComplexFloat>		ComplexDRef;
typedef DeviceRef<float>			FloatDRef;


/*
* Memcpy
*/
template<typename T1, typename T2>
void MemcpyA(T1& dst, const T2& src, size_t size, cudaStream_t stream)
{
	//typename T1::Type T1;
	//typename T2::Type;

	Assert(sizeof(typename T1::Type) == sizeof(typename T2::Type));
	checkCudaErrors(
		cudaMemcpyAsync(*dst,
				*src,
				size * sizeof(typename T1::Type),
				GetMemoryDirection(dst,src),
				stream));
};

//copy ALL data (obtains size from src);
template<typename T1, typename T2>
void MemcpyA(T1& dst, const T2& src, cudaStream_t stream)
{
	Assert(sizeof(T1::Type) == sizeof(T2::Type));
	checkCudaErrors(
		cudaMemcpyAsync(*dst,
                        *src,
                        src.size() * sizeof(typename T1::Type),
                        GetMemoryDirection(dst,src),stream));
};



/**
* @param dir: one of {D2H, H2D}
*/
template<typename T1, typename T2>
void MemcpyA(T1 dst, const T2 src, size_t size, MemcpyDir dir, cudaStream_t stream) 
{
	checkCudaErrors(cudaMemcpyAsync(dst, src, size * sizeof(*src), toDir(dir), stream));
};

#endif
