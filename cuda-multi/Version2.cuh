#ifndef CUDA_MULTI_VERSION_2_CUH
#define CUDA_MULTI_VERSION_2_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace multi
{

class KernelArgsCoalesced : public KernelArgsBase<KernelArgsValues>
{

public:

    KernelArgsCoalesced(KernelArgsValues &v) : KernelArgsBase(v) { }
    


    __device__ inline void setAlphaAt(const int optionIdx, const int optionCount, const int index, const real value) override
    {
        values.alphas[optionCount * index + optionIdx] = value;
    }

    __device__ inline real getAlphaAt(const int optionIdx, const int optionCount, const int index) override
    {
        return values.alphas[optionCount * index + optionIdx];
    }

};

class KernelRunCoalesced : public KernelRunBase
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

        runKernel<KernelArgsCoalesced>(cudaOptions, results, dInds, values);
    }
};

}

}

#endif
