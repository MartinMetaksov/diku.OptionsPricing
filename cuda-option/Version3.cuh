#ifndef CUDA_VERSION_3_CUH
#define CUDA_VERSION_3_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"
#include <thrust/transform_scan.h>

using namespace chrono;
using namespace trinom;

namespace cuda
{

class KernelArgsCoalescedBlock : public KernelArgsBase
{
private:
    int32_t *QsInds;
    int32_t *alphasInds;
    int widthBlockStartIndex;
    int heightBlockStartIndex;

public:

    KernelArgsCoalescedBlock(real *res, real *QsAll, real *QsCopyAll, real *alphasAll, int32_t *QsInds, int32_t *alphasInds)
        : KernelArgsBase(res, QsAll, QsCopyAll, alphasAll)
    {
        this->QsInds = QsInds;
        this->alphasInds = alphasInds;
    }

    __device__ inline void init(const CudaOptions &options) override
    {
        widthBlockStartIndex = blockIdx.x == 0 ? 0 : QsInds[blockIdx.x - 1];
        heightBlockStartIndex = blockIdx.x == 0 ? 0 : alphasInds[blockIdx.x - 1];
    }

    __device__ void fillQs(const int count, const int value) override
    {
        auto ptr = QsAll + widthBlockStartIndex + threadIdx.x;

        for (auto i = 0; i < count; ++i)
        {
            *ptr = value;
            ptr += blockDim.x;
        }
    }

    __device__ inline void setQAt(const int index, const real value) override
    {
        QsAll[widthBlockStartIndex + blockDim.x * index + threadIdx.x] = value;
    }

    __device__ inline void setQCopyAt(const int index, const real value) override
    {
        QsCopyAll[widthBlockStartIndex + blockDim.x * index + threadIdx.x] = value;
    }

    __device__ inline void setAlphaAt(const int index, const real value) override
    {
        alphasAll[heightBlockStartIndex + blockDim.x * index + threadIdx.x] = value;
    }

    __device__ inline void setResult(const int jmax) override
    {
        res[getIdx()] = QsAll[widthBlockStartIndex + blockDim.x * jmax + threadIdx.x];
    }

    __device__ inline real getQAt(const int index) const override { return QsAll[widthBlockStartIndex + blockDim.x * index + threadIdx.x]; }

    __device__ inline real getAlphaAt(const int index) const override { return alphasAll[heightBlockStartIndex + blockDim.x * index + threadIdx.x]; }
};

struct same_block_indices
{
    const int32_t BlockSize;

    same_block_indices(int32_t blockSize) : BlockSize(blockSize) {}

    __host__ __device__ bool operator()(const int32_t &lhs, const int32_t &rhs) const {return lhs / BlockSize == rhs / BlockSize;}
};

struct times_block_size
{
    const int32_t BlockSize;

    times_block_size(int32_t blockSize) : BlockSize(blockSize) {}

    __host__ __device__ int32_t operator()(const int32_t &x) const {return x * BlockSize;}
};

void computeOptionsCoalescedBlock(const Options &options, const Yield &yield, vector<real> &results, 
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
    
    // Create block indices.
    const auto blockCount = ceil(options.N / ((float)blockSize));
    thrust::device_vector<int32_t> keys(options.N);
    thrust::sequence(keys.begin(), keys.end());

    thrust::device_vector<int32_t> QsInds(blockCount);
    thrust::device_vector<int32_t> alphasInds(blockCount);
    thrust::device_vector<int32_t> keysOut(blockCount);
    thrust::reduce_by_key(keys.begin(), keys.end(), widths.begin(), keysOut.begin(), QsInds.begin(), same_block_indices(blockSize), thrust::maximum<int32_t>());
    thrust::reduce_by_key(keys.begin(), keys.end(), heights.begin(), keysOut.begin(), alphasInds.begin(), same_block_indices(blockSize), thrust::maximum<int32_t>());

    thrust::transform_inclusive_scan(QsInds.begin(), QsInds.end(), QsInds.begin(), times_block_size(blockSize), thrust::plus<int32_t>());
    thrust::transform_inclusive_scan(alphasInds.begin(), alphasInds.end(), alphasInds.begin(), times_block_size(blockSize), thrust::plus<int32_t>());

    const int totalQsCount = QsInds[blockCount - 1];
    const int totalAlphasCount = alphasInds[blockCount - 1];

    thrust::device_vector<real> Qs(totalQsCount);
    thrust::device_vector<real> QsCopy(totalQsCount);
    thrust::device_vector<real> alphas(totalAlphasCount);
    thrust::device_vector<real> result(options.N);

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
    auto d_QsInds = thrust::raw_pointer_cast(QsInds.data());
    auto d_alphasInds = thrust::raw_pointer_cast(alphasInds.data());
    KernelArgsCoalescedBlock kernelArgs(d_result, d_Qs, d_QsCopy, d_alphas, d_QsInds, d_alphasInds);

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
