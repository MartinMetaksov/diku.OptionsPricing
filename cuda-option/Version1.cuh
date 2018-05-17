#ifndef CUDA_VERSION_1_CUH
#define CUDA_VERSION_1_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

using namespace chrono;
using namespace trinom;

namespace cuda
{

class KernelArgsNaive : public KernelArgsBase
{
private:
    real *Qs;
    real *QsCopy;
    real *alphas;

public:

    KernelArgsNaive(real *res, real *QsAll, real *QsCopyAll, real *alphasAll)
        : KernelArgsBase(res, QsAll, QsCopyAll, alphasAll)
    { }

    __device__ inline void init(const CudaOptions &options) override
    {
        auto idx = getIdx();
        auto QsInd = idx == 0 ? 0 : options.Widths[idx - 1];
        auto alphasInd = idx == 0 ? 0 : options.Heights[idx - 1];
        Qs = QsAll + QsInd;
        QsCopy = QsCopyAll + QsInd;
        alphas = alphasAll + alphasInd;
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
        res[getIdx()] = Qs[jmax];
    }

    __device__ inline real getQAt(const int index) const override { return Qs[index]; }

    __device__ inline real getAlphaAt(const int index) const override { return alphas[index]; }
};

void computeOptionsNaive(const Options &options, const Yield &yield, vector<real> &results, 
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

    // Compute indices.
    thrust::inclusive_scan(widths.begin(), widths.end(), widths.begin());
    thrust::inclusive_scan(heights.begin(), heights.end(), heights.begin());

    // Allocate temporary vectors.
    const int totalQsCount = widths[options.N - 1];
    const int totalAlphasCount = heights[options.N - 1];
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
    KernelArgsNaive kernelArgs(d_result, d_Qs, d_QsCopy, d_alphas);

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