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

    void init(const Args &args, vector<OptionConstants> &optionConstants, int &yieldSize)
    {
        // Read options from filename, allocate the result array
        auto options = Option::readOptions(args.options);
        auto yield = Yield::readYieldCurve(args.yield);
        yieldSize = yield.size();

        optionConstants.reserve(options.size());
        for (auto &option : options)
        {
            auto constant = OptionConstants::computeConstants(option);
            optionConstants.push_back(constant);
        }

        OptionConstants::sortConstants(optionConstants, args.sort, args.test);
        CudaSafeCall(cudaMemcpyToSymbol(YieldCurve, yield.data(), yield.size() * sizeof(Yield)));
    }

}
#endif
