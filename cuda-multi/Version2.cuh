#ifndef CUDA_MULTI_VERSION_2_CUH
#define CUDA_MULTI_VERSION_2_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace multi
{

struct KernelArgsValuesCoalesced
{
    real *res;
    real *alphas;
    int32_t *inds;
    int32_t maxHeight;
};

class KernelArgsCoalesced : public KernelArgsBase<KernelArgsValuesCoalesced>
{

private:

    int optionIdx;
    int optionCount;

public:

    KernelArgsCoalesced(KernelArgsValuesCoalesced &v) : KernelArgsBase(v) { }
    
    __device__ inline void init(const int optionIdxBlock, const int idxBlock, const int idxBlockNext, const int optionCount)
    {
        this->optionIdx = idxBlock + optionIdxBlock;
        this->optionCount = optionCount;
    }

    __device__ inline void setAlphaAt(const int index, const real value) override
    {
        values.alphas[optionCount * index + optionIdx] = value;
    }

    __device__ inline real getAlphaAt(const int index) const override
    {
        return values.alphas[optionCount * index + optionIdx];
    }

    __device__ inline int getMaxHeight() const override
    {
        return values.maxHeight;
    }

    __device__ inline int getOptionIdx() const override
    {
        return optionIdx;
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

        KernelArgsValuesCoalesced values;

        // Get the max height
        values.maxHeight = thrust::max_element(heights.begin(), heights.end())[0];
        const int totalAlphasCount = cudaOptions.N * values.maxHeight;

        runKernel<KernelArgsCoalesced>(cudaOptions, results, dInds, values, totalAlphasCount);
    }
};

}

}

#endif
