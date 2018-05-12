#ifndef CUDA_VERSION_3_CUH
#define CUDA_VERSION_3_CUH

#include "../cuda/CudaDomain.cuh"
#include <thrust/transform_scan.h>

using namespace chrono;
using namespace trinom;

namespace cuda
{

__device__ inline real* getUnpaddedArrayAt(const int index, real *array, const int threadId, const int blockSize, const int blockStart)
{
    return array + blockStart + blockSize * index + threadId;
}

__device__ void fillUnpaddedArrayColumn(const int count, const real value, real *array, const int threadId, const int blockSize, const int blockStart)
{
    auto ptr = getUnpaddedArrayAt(0, array, threadId, blockSize, blockStart);
    for (auto i = 0; i < count; ++i)
    {
        *ptr = value;
        ptr += blockSize;
    }
}

__global__ void
kernelPaddingPerThreadBlock(const CudaOptions options, real *res, real *QsAll, real *QsCopyAll, real *alphasAll, int32_t *QsInds, int32_t *alphasInds)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Out of options check
    if (idx >= options.N) return;
    
    const int widthBlockStartIndex = blockIdx.x == 0 ? 0 : QsInds[blockIdx.x - 1];
    const int heightBlockStartIndex = blockIdx.x == 0 ? 0 : alphasInds[blockIdx.x - 1];

    OptionConstants c;
    computeConstants(c, options, idx);

    auto alpha = getYieldAtYear(c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize);
    *getUnpaddedArrayAt(c.jmax, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) = one;
    *getUnpaddedArrayAt(0, alphasAll, threadIdx.x, blockDim.x, heightBlockStartIndex) = alpha;

    for (auto i = 1; i <= c.n; ++i)
    {
        const auto jhigh = min(i, c.jmax);
        real alpha_val = 0;

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - (-c.jmax); // array index for j
            
            auto expp1 = j == jhigh ? zero : *getUnpaddedArrayAt(jind + 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) * exp(-(alpha + (j + 1) * c.dr) * c.dt);
            auto expm = *getUnpaddedArrayAt(jind, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) * exp(-(alpha + j * c.dr) * c.dt);
            auto expm1 = j == -jhigh ? zero : *getUnpaddedArrayAt(jind - 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) * exp(-(alpha + (j - 1) * c.dr) * c.dt);
            real Q;

            if (i == 1) {
                if (j == -jhigh) {
                    Q = computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                } else if (j == jhigh) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1;
                } else {
                    Q = computeJValue(j, c.jmax, c.M, 2) * expm;
                }
            }
            else if (i <= c.jmax) {
                if (j == -jhigh) {
                    Q = computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                } else if (j == -jhigh + 1) {
                    Q = computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                } else if (j == jhigh) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1;
                } else if (j == jhigh - 1) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm;
                } else {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                }
            } else {
                if (j == -jhigh) {
                    Q = computeJValue(j, c.jmax, c.M, 3) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                } else if (j == -jhigh + 1) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 2) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                            
                } else if (j == jhigh) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 1) * expm;
                } else if (j == jhigh - 1) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 2) * expp1;
                            
                } else {
                    Q = ((j == -jhigh + 2) ? computeJValue(j - 2, c.jmax, c.M, 1) * *getUnpaddedArrayAt(jind - 2, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) * exp(-(alpha + (j - 2) * c.dr) * c.dt) : zero) +
                        computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1 +
                        ((j == jhigh - 2) ? computeJValue(j + 2, c.jmax, c.M, 3) * *getUnpaddedArrayAt(jind + 2, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) * exp(-(alpha + (j + 2) * c.dr) * c.dt) : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            *getUnpaddedArrayAt(jind, QsCopyAll, threadIdx.x, blockDim.x, widthBlockStartIndex) = Q;
            alpha_val += Q * exp(-j * c.dr * c.dt);
        }

        alpha = computeAlpha(alpha_val, i-1, c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize);
        *getUnpaddedArrayAt(i, alphasAll, threadIdx.x, blockDim.x, heightBlockStartIndex) = alpha;

        // Switch Qs
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;
        fillUnpaddedArrayColumn(c.width, 0, QsCopyAll, threadIdx.x, blockDim.x, widthBlockStartIndex);
    }
    
    // Backward propagation
    fillUnpaddedArrayColumn(c.width, 100, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex);


    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = *getUnpaddedArrayAt(i, alphasAll, threadIdx.x, blockDim.x, heightBlockStartIndex);
        
        auto isMaturity = i == ((int)(c.t / c.dt));

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            auto callExp = exp(-(alpha + j * c.dr) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * *getUnpaddedArrayAt(jind, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(j, c.jmax, c.M, 2) * *getUnpaddedArrayAt(jind - 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(j, c.jmax, c.M, 3) * *getUnpaddedArrayAt(jind - 2, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex)) *
                      callExp;
            }
            else if (j == -c.jmax)
            {
                // Bottom edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * *getUnpaddedArrayAt(jind + 2, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(j, c.jmax, c.M, 2) * *getUnpaddedArrayAt(jind + 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(j, c.jmax, c.M, 3) * *getUnpaddedArrayAt(jind, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex)) *
                      callExp;
            }
            else
            {
                // Standard branching
                res = (computeJValue(j, c.jmax, c.M, 1) * *getUnpaddedArrayAt(jind + 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(j, c.jmax, c.M, 2) * *getUnpaddedArrayAt(jind, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(j, c.jmax, c.M, 3) * *getUnpaddedArrayAt(jind - 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex)) *
                      callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            *getUnpaddedArrayAt(jind, QsCopyAll, threadIdx.x, blockDim.x, widthBlockStartIndex) = computeCallValue(isMaturity, c, res);
        }

        // Switch call arrays
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;

        fillUnpaddedArrayColumn(c.width, 0, QsCopyAll, threadIdx.x, blockDim.x, widthBlockStartIndex);
    }

    res[idx] = *getUnpaddedArrayAt(c.jmax, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex);
}

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

void computeOptionsWithPaddingPerThreadBlock(const Options &options, const Yield &yield, vector<real> &results, 
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

    auto time_begin_kernel = steady_clock::now();
    kernelPaddingPerThreadBlock<<<blockCount, blockSize>>>(cudaOptions, d_result, d_Qs, d_QsCopy, d_alphas, d_QsInds, d_alphasInds);
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
