#ifndef CUDA_DOMAIN_CUH
#define CUDA_DOMAIN_CUH

#include "../common/OptionConstants.hpp"
#include "../common/Domain.hpp"
#include "../common/Arrays.hpp"
#include "../common/Args.hpp"
#include "../cuda/CudaErrors.cuh"
#include <cuda_runtime.h>
#include <chrono>

using namespace trinom;

namespace cuda
{
    __constant__ Yield YieldCurve[100];

}
#endif
