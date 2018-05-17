#ifndef CUDA_VERSION_2_CUH
#define CUDA_VERSION_2_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"
#include <thrust/extrema.h>

using namespace chrono;
using namespace trinom;

namespace cuda
{

class KernelArgsCoalesced : public KernelArgsBase
{
private:
    int N;

public:

    KernelArgsCoalesced(real *res, real *QsAll, real *QsCopyAll, real *alphasAll)
        : KernelArgsBase(res, QsAll, QsCopyAll, alphasAll)
    { }

    __device__ inline void init(const CudaOptions &options) override
    {
        N = options.N;
    }

    __device__ void fillQs(const int count, const int value) override
    {
        auto ptr = QsAll + getIdx();

        for (auto i = 0; i < count; ++i)
        {
            *ptr = value;
            ptr += N;
        }
    }

    __device__ inline void setQAt(const int index, const real value) override
    {
        QsAll[index * N + getIdx()] = value;
    }

    __device__ inline void setQCopyAt(const int index, const real value) override
    {
        QsCopyAll[index * N + getIdx()] = value;
    }

    __device__ inline void setAlphaAt(const int index, const real value) override
    {
        alphasAll[index * N + getIdx()] = value;
    }

    __device__ inline void setResult(const int jmax) override
    {
        res[getIdx()] = QsAll[jmax * N + getIdx()];
    }

    __device__ inline real getQAt(const int index) const override { return QsAll[index * N + getIdx()]; }

    __device__ inline real getAlphaAt(const int index) const override { return alphasAll[index * N + getIdx()]; }
};

void computeOptionsCoalesced(const Options &options, const Yield &yield, vector<real> &results, 
    const int blockSize = 64, const SortType sortType = SortType::NONE, bool isTest = false)
{
    size_t memoryFreeStart, memoryFree, memoryTotal;
    cudaMemGetInfo(&memoryFreeStart, &memoryTotal);

    thrust::device_vector<uint16_t> strikePrices(options.StrikePrices.begin(), options.StrikePrices.end());
    thrust::device_vector<uint16_t> maturities(options.Maturities.begin(), options.Maturities.end());
    thrust::device_vector<uint16_t> lengths(options.Lengths.begin(), options.Lengths.end());
    thrust::device_vector<uint16_t> termUnits(options.TermUnits.begin(), options.TermUnits.end());
    thrust::device_vector<uint16_t> termStepCounts(options.TermStepCounts.begin(), options.TermStepCounts.end());
    thrust::device_vector<real> reversionRates(options.ReversionRates.begin(), options.ReversionRates.end());
    thrust::device_vector<real> volatilities(options.Volatilities.begin(), options.Volatilities.end());
    thrust::device_vector<OptionType> types(options.Types.begin(), options.Types.end());

    thrust::device_vector<real> yieldPrices(yield.Prices.begin(), yield.Prices.end());
    thrust::device_vector<int32_t> yieldTimeSteps(yield.TimeSteps.begin(), yield.TimeSteps.end());

    thrust::device_vector<int32_t> widths(options.N);
    thrust::device_vector<int32_t> heights(options.N);

    CudaOptions cudaOptions(options, yield.N, sortType, isTest, strikePrices, maturities, lengths, termUnits, 
        termStepCounts, reversionRates, volatilities, types, yieldPrices, yieldTimeSteps, widths, heights);
    
    // Compute padding
    int maxWidth = thrust::max_element(widths.begin(), widths.end())[0];
    int maxHeight = thrust::max_element(heights.begin(), heights.end())[0];
    int totalQsCount = options.N * maxWidth;
    int totalAlphasCount = options.N * maxHeight;

    thrust::device_vector<real> Qs(totalQsCount);
    thrust::device_vector<real> QsCopy(totalQsCount);
    thrust::device_vector<real> alphas(totalAlphasCount);
    thrust::device_vector<real> result(options.N);
    
    const auto blockCount = ceil(options.N / ((float)blockSize));

    if (isTest)
    {
        cout << "Running trinomial option pricing for " << options.N << " options with block size " << blockSize << endl;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&memoryFree, &memoryTotal);
        cout << "Memory used " << (memoryFreeStart - memoryFree) / (1024.0 * 1024.0) << " MB out of " << memoryTotal / (1024.0 * 1024.0) << " MB " << endl;
    }

    auto d_result = thrust::raw_pointer_cast(result.data());
    auto d_Qs = thrust::raw_pointer_cast(Qs.data());
    auto d_QsCopy = thrust::raw_pointer_cast(QsCopy.data());
    auto d_alphas = thrust::raw_pointer_cast(alphas.data());
    KernelArgsCoalesced kernelArgs(d_result, d_Qs, d_QsCopy, d_alphas);

    auto time_begin_kernel = steady_clock::now();
    kernelOneOptionPerThread<<<blockCount, blockSize>>>(cudaOptions, kernelArgs);
    cudaThreadSynchronize();
    auto time_end_kernel = steady_clock::now();
    if (isTest)
    {
        cout << "Kernel executed in " << duration_cast<microseconds>(time_end_kernel - time_begin_kernel).count() << " microsec" << endl;
    }

    CudaCheckError();

    // Copy result
    thrust::copy(result.begin(), result.end(), results.begin());
}

}

#endif