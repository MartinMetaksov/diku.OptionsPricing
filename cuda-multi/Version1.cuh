#ifndef CUDA_MULTI_VERSION_1_CUH
#define CUDA_MULTI_VERSION_1_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace multi
{

class KernelArgsNaive : public KernelArgsBase<KernelArgsValues>
{

public:

    KernelArgsNaive(KernelArgsValues &v) : KernelArgsBase(v) { }

    __device__ inline void setAlphaAt(const int optionIdx, const int index, const real value) override
    {
        values.alphas[values.maxHeight * optionIdx + index] = value;
    }

    __device__ inline real getAlphaAt(const int optionIdx, const int index) override
    {
        return values.alphas[values.maxHeight * optionIdx + index];
    }
};

class KernelRunNaive : public KernelRunBase
{

protected:
    void runPreprocessing(CudaOptions &cudaOptions, std::vector<real> &results,
        thrust::device_vector<int32_t> &widths, thrust::device_vector<int32_t> &heights) override
    {
        // Compute indices.
        thrust::host_vector<int32_t> hostWidths = widths;
        thrust::host_vector<int32_t> hInds;

        auto counter = 0;
        for (auto i = 0; i < cudaOptions.N; ++i)
        {
            auto w = hostWidths[i];
            counter += w;
            if (counter > blockSize)
            {
                hInds.push_back(i);
                counter = w;
            }
        }
        hInds.push_back(cudaOptions.N);

        thrust::device_vector<int32_t> dInds = hInds;

        KernelArgsValues values;

        runKernel<KernelArgsNaive>(cudaOptions, results, dInds, values);
    }
};

}

}

#endif
