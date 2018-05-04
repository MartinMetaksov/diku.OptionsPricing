#ifndef CUDA_INTEROP_H
#define CUDA_INTEROP_H

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT const
#endif

#ifdef __CUDA_ARCH__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifdef __CUDA_ARCH__
#define HOST __host__
#else
#define HOST
#endif

#endif
