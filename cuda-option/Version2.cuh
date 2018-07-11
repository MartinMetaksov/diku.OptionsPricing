#ifndef CUDA_VERSION_2_CUH
#define CUDA_VERSION_2_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace option
{

class KernelArgsCoalesced : public KernelArgsBase<KernelArgsValues>
{
private:
    int N;

public:

    KernelArgsCoalesced(KernelArgsValues &v) : KernelArgsBase(v) { }

    __device__ inline void init(const KernelOptions &options) override
    {
        N = options.N;
    }

    __device__ void fillQs(const int count, const int value) override
    {
        auto ptr = values.QsAll + getIdx();

        for (auto i = 0; i < count; ++i)
        {
            *ptr = value;
            ptr += N;
        }
    }

    __device__ inline void setQAt(const int index, const real value) override
    {
        values.QsAll[index * N + getIdx()] = value;
    }

    __device__ inline void setQCopyAt(const int index, const real value) override
    {
        values.QsCopyAll[index * N + getIdx()] = value;
    }

    __device__ inline void setAlphaAt(const int index, const real value) override
    {
        values.alphasAll[index * N + getIdx()] = value;
    }

    __device__ inline void setResult(const int jmax) override
    {
        values.res[getIdx()] = values.QsAll[jmax * N + getIdx()];
    }

    __device__ inline real getQAt(const int index) const override { return values.QsAll[index * N + getIdx()]; }

    __device__ inline real getAlphaAt(const int index) const override { return values.alphasAll[index * N + getIdx()]; }
};

class KernelRunCoalesced : public KernelRunBase
{

protected:
    void runPreprocessing(CudaOptions &options, std::vector<real> &results) override
    {
        // Compute padding
        int maxWidth = thrust::max_element(options.Widths.begin(), options.Widths.end())[0];
        int maxHeight = thrust::max_element(options.Heights.begin(), options.Heights.end())[0];
        int totalQsCount = options.N * maxWidth;
        int totalAlphasCount = options.N * maxHeight;
        KernelArgsValues values;

        runKernel<KernelArgsCoalesced>(options, results, totalQsCount, totalAlphasCount, values);
    }
};

}

}

#endif
