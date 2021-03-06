#ifndef CUDA_MULTI_VERSION_1_CUH
#define CUDA_MULTI_VERSION_1_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace multi
{

struct KernelArgsValuesNaive
{
    real *res;
    real *alphas;
    int32_t *inds;
    int32_t maxHeight;
};

class KernelArgsNaive : public KernelArgsBase<KernelArgsValuesNaive>
{

private:
    
    int optionIdx;
    int optionCount;

public:

    KernelArgsNaive(KernelArgsValuesNaive &v) : KernelArgsBase(v) { }

    __device__ inline void init(const int optionIdxBlock, const int idxBlock, const int idxBlockNext, const int optionCount)
    {
        this->optionIdx = idxBlock + optionIdxBlock;
        this->optionCount = optionCount;
    }

    __device__ inline void setAlphaAt(const int index, const real value) override
    {
        values.alphas[values.maxHeight * optionIdx + index] = value;
    }

    __device__ inline real getAlphaAt(const int index) const override
    {
        return values.alphas[values.maxHeight * optionIdx + index];
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

class KernelRunNaive : public KernelRunBase
{

protected:
    void runPreprocessing(CudaOptions &options, std::vector<real> &results) override
    {
        // Compute indices.
        thrust::host_vector<int32_t> hostWidths = options.Widths;
        thrust::host_vector<int32_t> hInds;

        auto counter = 0;
        for (auto i = 0; i < options.N; ++i)
        {
            auto w = hostWidths[i];
            counter += w;
            if (counter > BlockSize)
            {
                hInds.push_back(i);
                counter = w;
            }
        }
        hInds.push_back(options.N);

        thrust::device_vector<int32_t> dInds = hInds;

        KernelArgsValuesNaive values;

        // Get the max height
        values.maxHeight = thrust::max_element(options.Heights.begin(), options.Heights.end())[0];
        const int totalAlphasCount = options.N * values.maxHeight;

        runKernel<KernelArgsNaive>(options, results, dInds, values, totalAlphasCount);
    }
};

}

}

#endif
