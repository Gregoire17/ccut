#ifndef _CCUT_MEMORY_IMPL_H_
#define _CCUT_MEMORY_IMPL_H_

#include <boost/shared_ptr.hpp>

#include <driver_types.h>
#include <complex_t.h>

#include <ccut/utils.cuh>
#include <utilities.h>

#include <non_copyable.h>
#include <Assertion.h>
#include <log.h>

struct Host{};
struct Device{};

template<typename T>
struct HostAlloc : public Host, NonCopyable
{
	static cudaError_t Alloc(T** ptr, size_t size)
	{
		//TODO change to alloc with cufft malloc
		return cudaMallocHost(ptr, size*sizeof(T));	
	}

	static cudaError_t Release(T* ptr)
	{
		return cudaFreeHost(ptr);	
	}

	static void Memset(T* ptr, int value, size_t size)
	{
		memset(ptr, value, size*sizeof(T));
	}
};

template<typename T>
struct DeviceAlloc : public Device, NonCopyable
{
	static cudaError_t Alloc(T** ptr, size_t size)
	{
		return cudaMalloc(ptr, size*sizeof(T));	
	}

	static cudaError_t Release(T* ptr)
	{
		return cudaFree(ptr);	
	}

	static void MemsetA(T* ptr, int value, size_t size, cudaStream_t stream)
	{
		checkCudaErrors(cudaMemsetAsync(ptr, value, size*sizeof(T), stream));
	}
};


template<typename T, typename AllocPolicy>
class Memory : public AllocPolicy{
public:
	typedef T Type;

	Memory(size_t size) :
		m_ptr(0),
		m_size(size)
	{
		checkCudaErrors(AllocPolicy::Alloc(&m_ptr,size));
	}

	/*virtual*/ T* operator*() const{return m_ptr;}

	virtual ~Memory()
	{
		checkCudaErrors(AllocPolicy::Release(m_ptr));
	}

	void MemsetA(cudaStream_t stream, int value = 0)
	{
		AllocPolicy::MemsetA(m_ptr, value, m_size, stream);
	}

	size_t size() const {return m_size;}

private:
	T*  m_ptr;
	size_t m_size;
};

template<typename T>
struct MemoryHost : public Memory<T, HostAlloc<T> >
{
	MemoryHost(size_t size) : Memory<T,HostAlloc<T> >(size) {};
};

template<typename T>
struct MemoryDevice : public Memory<T, DeviceAlloc<T> >
{
	MemoryDevice(size_t size) : Memory<T,DeviceAlloc<T> >(size) {};
};


/*
* Pointers - without allocation and release
*/
template<typename T>
struct HostRef : public Host
{
	typedef T Type;

	HostRef(T* ptr) :
		m_ptr(ptr)
	{};

	/*virtual*/ T* operator*() const{return m_ptr;}

private:
	T* m_ptr;
};

template<typename T>
struct DeviceRef : public Device
{
	typedef T Type;

	DeviceRef(T* ptr) :
		m_ptr(ptr)
	{};

	/*virtual*/ T* operator*() const{return m_ptr;}

private:
	T* m_ptr;
};



//Memory Direction
inline cudaMemcpyKind GetMemoryDirection(const Device& /*dst*/, const Host& /*src*/ )
{
		return cudaMemcpyHostToDevice;
};

inline cudaMemcpyKind GetMemoryDirection(const Host& /*dst*/, const Device& /*src*/)
{
	return cudaMemcpyDeviceToHost;
};

inline cudaMemcpyKind GetMemoryDirection(const Device& /*dst*/, const Device& /*src*/)
{
	return cudaMemcpyDeviceToDevice;
};

enum MemcpyDir
{
	D2H,
	H2D,
	D2D
};


inline cudaMemcpyKind toDir(MemcpyDir dir)
{
	switch(dir)
	{
	case D2H:
		return cudaMemcpyDeviceToHost;
	case H2D:
		return cudaMemcpyHostToDevice;
	case D2D:
		return cudaMemcpyDeviceToDevice;
	default:
		LogCritical("Invalid direction");
		Assert(0);
		return cudaMemcpyHostToDevice;
	}
}


#endif
