#ifndef CUDA_INTEROP_H
#define CUDA_INTEROP_H

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#define DEVICE __device__
#define HOST __host__
#define MAX(X, Y) max((X), (Y))
#else
#define CONSTANT const
#define DEVICE
#define HOST
#define MAX(X, Y) std::max((X), (Y))
#endif

#endif
