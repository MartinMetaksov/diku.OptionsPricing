#ifndef CUDA_VERSION_1_CUH
#define CUDA_VERSION_1_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace option
{

class KernelArgsNaive : public KernelArgsBase<KernelArgsValues>
{
private:
    real *Qs;
    real *QsCopy;
    real *alphas;

public:

    KernelArgsNaive(KernelArgsValues &v) : KernelArgsBase(v) { }

    __device__ inline void init(const CudaOptions &options) override
    {
        auto idx = getIdx();
        auto QsInd = idx == 0 ? 0 : options.Widths[idx - 1];
        auto alphasInd = idx == 0 ? 0 : options.Heights[idx - 1];
        Qs = values.QsAll + QsInd;
        QsCopy = values.QsCopyAll + QsInd;
        alphas = values.alphasAll + alphasInd;
    }

    __device__ inline void switchQs()
    {
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;
    }

    __device__ void fillQs(const int count, const int value) override
    {
        for (auto i = 0; i < count; ++i)
        {
            Qs[i] = value;
        }
    }

    __device__ inline void setQAt(const int index, const real value) override
    {
        Qs[index] = value;
    }

    __device__ inline void setQCopyAt(const int index, const real value) override
    {
        QsCopy[index] = value;
    }

    __device__ inline void setAlphaAt(const int index, const real value) override
    {
        alphas[index] = value;
    }

    __device__ inline void setResult(const int jmax) override
    {
        values.res[getIdx()] = Qs[jmax];
    }

    __device__ inline real getQAt(const int index) const override { return Qs[index]; }

    __device__ inline real getAlphaAt(const int index) const override { return alphas[index]; }
};

class KernelRunNaive : public KernelRunBase
{

protected:
    void runPreprocessing(CudaOptions &cudaOptions, std::vector<real> &results,
        thrust::device_vector<int32_t> &widths, thrust::device_vector<int32_t> &heights) override
    {
        // Compute indices.
        thrust::inclusive_scan(widths.begin(), widths.end(), widths.begin());
        thrust::inclusive_scan(heights.begin(), heights.end(), heights.begin());

        // Allocate temporary vectors.
        const int totalQsCount = widths[cudaOptions.N - 1];
        const int totalAlphasCount = heights[cudaOptions.N - 1];
        KernelArgsValues values;

        runKernel<KernelArgsNaive>(cudaOptions, results, totalQsCount, totalAlphasCount, values);
    }
};

}

}

#endif
