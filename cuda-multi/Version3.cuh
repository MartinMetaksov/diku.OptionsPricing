#ifndef CUDA_MULTI_VERSION_3_CUH
#define CUDA_MULTI_VERSION_3_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace multi
{  

struct KernelArgsValuesCoalescedBlock
{
    real *res;
    real *alphas;
    int32_t *inds;
    int32_t *alphaInds;
};

class KernelArgsCoalescedBlock : public KernelArgsBase<KernelArgsValuesCoalescedBlock>
{

private:

    int optionIdx;
    int alphaIdx;
    int maxHeight;
    int optionCountBlock;

public:

    KernelArgsCoalescedBlock(KernelArgsValuesCoalescedBlock &v) : KernelArgsBase(v) { }
    
    __device__ inline void init(const int optionIdxBlock, const int idxBlock, const int idxBlockNext, const int optionCount)
    {
        optionIdx = idxBlock + optionIdxBlock;
        optionCountBlock = idxBlockNext - idxBlock;
        auto alphaIdxBlock = (blockIdx.x == 0 ? 0 : values.alphaInds[blockIdx.x - 1]);
        maxHeight = (values.alphaInds[blockIdx.x] - alphaIdxBlock) / optionCountBlock;
        alphaIdx = alphaIdxBlock + optionIdxBlock;
    }

    __device__ inline void setAlphaAt(const int index, const real value) override
    {
        values.alphas[alphaIdx + optionCountBlock * index] = value;
    }

    __device__ inline real getAlphaAt(const int index) const override
    {
        return values.alphas[alphaIdx + optionCountBlock * index];
    }

    __device__ inline int getMaxHeight() const override
    {
        return maxHeight;
    }

    __device__ inline int getOptionIdx() const override
    {
        return optionIdx;
    }
};

class KernelRunCoalescedBlock : public KernelRunBase
{

protected:
    void runPreprocessing(CudaOptions &cudaOptions, std::vector<real> &results,
        thrust::device_vector<int32_t> &widths, thrust::device_vector<int32_t> &heights) override
    {
        // Compute indices.
        thrust::host_vector<int32_t> hostWidths = widths;
        thrust::host_vector<int32_t> hostHeights = heights;
        thrust::host_vector<int32_t> hInds;
        thrust::host_vector<int32_t> hAlphaInds;

        auto counter = 0;
        auto maxHeightBlock = 0;
        for (auto i = 0; i < cudaOptions.N; ++i)
        {
            auto w = hostWidths[i];
            auto h = hostHeights[i];
            counter += w;
            if (counter > blockSize)
            {
                auto alphasBlock = maxHeightBlock * (i - (hInds.empty() ? 0 : hInds.back()));
                hAlphaInds.push_back((hAlphaInds.empty() ? 0 : hAlphaInds.back()) + alphasBlock);
                hInds.push_back(i);
                counter = w;
                maxHeightBlock = 0;
            }
            if (h > maxHeightBlock) {
                maxHeightBlock = h;
            }
        }
        auto alphasBlock = maxHeightBlock * (cudaOptions.N - (hInds.empty() ? 0 : hInds.back()));
        hAlphaInds.push_back((hAlphaInds.empty() ? 0 : hAlphaInds.back()) + alphasBlock);
        hInds.push_back(cudaOptions.N);

        thrust::device_vector<int32_t> dInds = hInds;
        thrust::device_vector<int32_t> dAlphaInds = hAlphaInds;
        auto totalAlphasCount = hAlphaInds.back();

        KernelArgsValuesCoalescedBlock values;
        values.alphaInds = thrust::raw_pointer_cast(dAlphaInds.data());

        if (isTest)
        {
            deviceMemory += vectorsizeof(dAlphaInds);
        }

        runKernel<KernelArgsCoalescedBlock>(cudaOptions, results, dInds, values, totalAlphasCount);
    }
};

}

}

#endif
