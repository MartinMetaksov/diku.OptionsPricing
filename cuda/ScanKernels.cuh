#ifndef SCAN_KERNELS_CUH
#define SCAN_KERNELS_CUH

#include <cuda_runtime.h>

namespace cuda
{

template <class T>
class Add
{
  public:
    typedef T BaseType;
    static __device__ __host__ inline T identity() { return (T)0; }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }
};

/***************************************/
/*** Scan Inclusive Helpers & Kernel ***/
/***************************************/
template <class OP, class T>
__device__ inline T scanIncWarp(volatile T *ptr, const unsigned int idx)
{
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)
        ptr[idx] = OP::apply(ptr[idx - 1], ptr[idx]);
    if (lane >= 2)
        ptr[idx] = OP::apply(ptr[idx - 2], ptr[idx]);
    if (lane >= 4)
        ptr[idx] = OP::apply(ptr[idx - 4], ptr[idx]);
    if (lane >= 8)
        ptr[idx] = OP::apply(ptr[idx - 8], ptr[idx]);
    if (lane >= 16)
        ptr[idx] = OP::apply(ptr[idx - 16], ptr[idx]);

    return const_cast<T &>(ptr[idx]);
}

template <class OP, class T>
__device__ inline T scanIncBlock(volatile T *ptr, const unsigned int idx = threadIdx.x)
{
    const unsigned int lane = idx & 31;
    const unsigned int warpid = idx >> 5;

    T val = scanIncWarp<OP, T>(ptr, idx);
    __syncthreads();

    // place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and
    //   max block size = 32^2 = 1024
    if (lane == 31)
    {
        ptr[warpid] = const_cast<T &>(ptr[idx]);
    }
    __syncthreads();

    //
    if (warpid == 0)
        scanIncWarp<OP, T>(ptr, idx);
    __syncthreads();

    if (warpid > 0)
    {
        val = OP::apply(ptr[warpid - 1], val);
    }
    __syncthreads();

    ptr[idx] = val;
    __syncthreads();
    
    return val;
}

/*************************************************/
/*************************************************/
/*** Segmented Inclusive Scan Helpers & Kernel ***/
/*************************************************/
/*************************************************/
template <class OP, class T, class F>
__device__ inline T sgmScanIncWarp(volatile T *ptr, volatile F *flg, const unsigned int idx)
{
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)
    {
        if (flg[idx] == 0)
        {
            ptr[idx] = OP::apply(ptr[idx - 1], ptr[idx]);
        }
        flg[idx] = flg[idx - 1] | flg[idx];
    }
    if (lane >= 2)
    {
        if (flg[idx] == 0)
        {
            ptr[idx] = OP::apply(ptr[idx - 2], ptr[idx]);
        }
        flg[idx] = flg[idx - 2] | flg[idx];
    }
    if (lane >= 4)
    {
        if (flg[idx] == 0)
        {
            ptr[idx] = OP::apply(ptr[idx - 4], ptr[idx]);
        }
        flg[idx] = flg[idx - 4] | flg[idx];
    }
    if (lane >= 8)
    {
        if (flg[idx] == 0)
        {
            ptr[idx] = OP::apply(ptr[idx - 8], ptr[idx]);
        }
        flg[idx] = flg[idx - 8] | flg[idx];
    }
    if (lane >= 16)
    {
        if (flg[idx] == 0)
        {
            ptr[idx] = OP::apply(ptr[idx - 16], ptr[idx]);
        }
        flg[idx] = flg[idx - 16] | flg[idx];
    }

    return const_cast<T &>(ptr[idx]);
}

template <class OP, class T, class F>
__device__ inline T sgmScanIncBlock(volatile T *ptr, volatile F *flg, const unsigned int idx = threadIdx.x)
{
    // const unsigned int lane = idx & 31;
    const unsigned int warpid = idx >> 5;
    const unsigned int warplst = (warpid << 5) + 31;

    // 1a: record whether this warp begins with an ``open'' segment.
    bool warp_is_open = (flg[(warpid << 5)] == 0);
    __syncthreads();

    // 1b: intra-warp segmented scan for each warp
    T val = sgmScanIncWarp<OP, T>(ptr, flg, idx);

    // 2a: the last value is the correct partial result
    T warp_total = const_cast<T &>(ptr[warplst]);

    // 2b: warp_flag is the OR-reduction of the flags
    //     in a warp, and is computed indirectly from
    //     the mindex in hd[]
    F warp_flag = flg[warplst] != 0 || !warp_is_open;
    bool will_accum = warp_is_open && (flg[idx] == 0);

    __syncthreads();

    // 2c: the last thread in a warp writes partial results
    //     in the first warp. Note that all fit in the first
    //     warp because warp = 32 and max block size is 32^2
    if (idx == warplst)
    {
        ptr[warpid] = warp_total; //ptr[idx];
        flg[warpid] = warp_flag;
    }
    __syncthreads();

    //
    if (warpid == 0)
        sgmScanIncWarp<OP, T>(ptr, flg, idx);
    __syncthreads();

    if (warpid > 0 && will_accum)
    {
        val = OP::apply(ptr[warpid - 1], val);
    }
    __syncthreads();

    ptr[idx] = val;
    __syncthreads();

    return val;
}

// // Sequential segmented scan implementation, useful for debugging.
// template<class T, class F>
// __device__ void sgmScanIncBlockSeq(T *values, F *flags)
// {
//     if (threadIdx.x == 0)
//     {
//         F counter = 0;
//         T scan = 0;
//         for (int i = 0; i < blockDim.x; ++i)
//         {
//             F flg = flags[i];
//             if (flg != 0)
//             {   
//                 if (counter > 0)
//                 {
//                     printf("sgmScanIncBlock: wrong flag at %d!\n", i);
//                 }
//                 counter = flg;
//                 scan = 0;
//             }

//             --counter;
//             scan += values[i];
//             values[i] = scan;
//         }
//         if (counter > 0)
//         {
//             printf("sgmScanIncBlock: wrong flag at the end!\n");
//         }
//     }
//     __syncthreads();
// }

// // Sequential scan implementation, useful for debugging.
// template<class T>
// __device__ void scanIncBlockSeq(T *values)
// {
//     if (threadIdx.x == 0)
//     {
//         T scan = 0;
//         for (int i = 0; i < blockDim.x; ++i)
//         {
//             scan += values[i];
//             values[i] = scan;
//         }
//     }
//     __syncthreads();
// }

}

#endif //SCAN_KERNELS_CUH
