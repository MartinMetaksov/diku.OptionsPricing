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
    void runPreprocessing(CudaOptions &options, std::vector<real> &results) override
    {
        // Compute indices.
        thrust::host_vector<int32_t> hostWidths = options.Widths;
        thrust::host_vector<int32_t> hostHeights = options.Heights;
        thrust::host_vector<int32_t> hInds;
        thrust::host_vector<int32_t> hAlphaInds;

        auto counter = 0;
        auto maxHeightBlock = 0;
<<<<<<< HEAD
        auto prevInd = 0;
        auto maxOptionsBlock = 0;
        for (auto i = 0; i < cudaOptions.N; ++i)
=======
        for (auto i = 0; i < options.N; ++i)
>>>>>>> master
        {
            auto w = hostWidths[i];
            auto h = hostHeights[i];

            counter += w;
            if (counter > BlockSize)
            {
                auto alphasBlock = maxHeightBlock * (i - (hInds.empty() ? 0 : hInds.back()));
                hAlphaInds.push_back((hAlphaInds.empty() ? 0 : hAlphaInds.back()) + alphasBlock);
                hInds.push_back(i);
                counter = w;
                maxHeightBlock = 0;

                auto optionsBlock = i - prevInd;
                if (optionsBlock > maxOptionsBlock) {
                    maxOptionsBlock = optionsBlock;
                }
                prevInd = i;
            }
            if (h > maxHeightBlock) {
                maxHeightBlock = h;
            }
        }
        auto alphasBlock = maxHeightBlock * (options.N - (hInds.empty() ? 0 : hInds.back()));
        hAlphaInds.push_back((hAlphaInds.empty() ? 0 : hAlphaInds.back()) + alphasBlock);
        hInds.push_back(options.N);

        auto optionsBlock = cudaOptions.N - prevInd;
        if (optionsBlock > maxOptionsBlock) {
            maxOptionsBlock = optionsBlock;
        }

        thrust::device_vector<int32_t> dInds = hInds;
        thrust::device_vector<int32_t> dAlphaInds = hAlphaInds;
        auto totalAlphasCount = hAlphaInds.back();

        KernelArgsValuesCoalescedBlock values;
        values.alphaInds = thrust::raw_pointer_cast(dAlphaInds.data());

        options.DeviceMemory += vectorsizeof(dAlphaInds);

<<<<<<< HEAD
        runKernel<KernelArgsCoalescedBlock>(cudaOptions, results, dInds, values, totalAlphasCount, maxOptionsBlock);
=======
        runKernel<KernelArgsCoalescedBlock>(options, results, dInds, values, totalAlphasCount);
>>>>>>> master
    }
};

}

}

#endif
