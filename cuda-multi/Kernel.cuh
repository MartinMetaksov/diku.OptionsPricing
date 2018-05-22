#ifndef CUDA_KERNEL_MULTI_CUH
#define CUDA_KERNEL_MULTI_CUH

#include "../cuda/CudaDomain.cuh"
#include "../cuda/ScanKernels.cuh"
#include <sstream>
#include <stdexcept>

using namespace trinom;

namespace cuda
{

namespace multi
{
        
struct KernelArgsValues
{
    real *res;
    real *alphas;
    int32_t *inds;
    int32_t maxHeight;
};

/**
Base class for kernel arguments.
Important! Don't call defined pure virtual functions within your implementation.
**/
template<class KernelArgsValuesT>
class KernelArgsBase
{

public:

    KernelArgsValuesT values;

    KernelArgsBase(KernelArgsValuesT &v) : values(v) { }

    __device__ virtual void setAlphaAt(const int optionIdx, const int index, const real value) = 0;

    __device__ virtual real getAlphaAt(const int optionIdx, const int index) = 0;
};

template<class KernelArgsT>
__global__ void kernelMultipleOptionsPerThreadBlock(const CudaOptions options, KernelArgsT args)
{
    extern __shared__ real sh_mem[];
    volatile real *Qs = (real *)&sh_mem;
    volatile int32_t *optionInds = (int32_t *) &sh_mem;     // Sharing the same array with Qs!
    volatile uint16_t *optionFlags = (uint16_t *) &Qs[blockDim.x];

    // Compute option indices and init Qs
    const auto idxBlock = blockIdx.x == 0 ? 0 : args.values.inds[blockIdx.x - 1];
    const auto idx = idxBlock + threadIdx.x;
    const auto nextIdx = args.values.inds[blockIdx.x];
    int width;
    if (idx < nextIdx)    // Don't fetch options from next block
    {
        width = options.Widths[idx];
        optionInds[threadIdx.x] = width;
    }
    else
    {
        optionInds[threadIdx.x] = 0;
    }
    __syncthreads();

    // Scan widths
    optionInds[threadIdx.x] = scanIncBlock<Add<int>>(optionInds, threadIdx.x);
    __syncthreads();
    
    int scannedWidthIdx = -1;
    if (idx <= nextIdx)
    {
        scannedWidthIdx = threadIdx.x == 0 ? 0 : optionInds[threadIdx.x - 1];
    }

    // Compute option indices
    optionInds[threadIdx.x] = 0;
    optionFlags[threadIdx.x] = 0;
    __syncthreads();

    if (idx < nextIdx)
    {
        optionInds[scannedWidthIdx] = threadIdx.x;
        optionFlags[scannedWidthIdx] = width;
    }
    else if (idx == nextIdx) // fake option to fill block
    {
        optionInds[scannedWidthIdx] = threadIdx.x;
        optionFlags[scannedWidthIdx] = blockDim.x - scannedWidthIdx;
    }
    __syncthreads();

    optionInds[threadIdx.x] = sgmScanIncBlock<Add<int>>(optionInds, optionFlags, threadIdx.x);
    __syncthreads();

    // Get the option and compute its constants
    OptionConstants c;
    const auto optionIdx = idxBlock + optionInds[threadIdx.x];
    if (optionIdx < nextIdx)
    {
        computeConstants(c, options, optionIdx);
    }

    // Let all threads know about their Q start
    if (idx <= nextIdx)
    {
        optionFlags[threadIdx.x] = scannedWidthIdx;
    }
    __syncthreads();
    scannedWidthIdx = optionFlags[optionInds[threadIdx.x]];

    if (optionIdx >= nextIdx)
    {
        c.n = 0;
        c.width = blockDim.x - scannedWidthIdx;
    }

    // Zero out Qs
    Qs[threadIdx.x] = 0;
    __syncthreads();

    // Set the initial alpha and Q values
    if (threadIdx.x == scannedWidthIdx && optionIdx < nextIdx)
    {
        args.setAlphaAt(optionIdx, 0, getYieldAtYear(c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize));
        Qs[scannedWidthIdx + c.jmax] = 1;    // Set starting Qs to 1$
    }
    __syncthreads();

    // Forward propagation
    for (int i = 1; i <= args.values.maxHeight; ++i)
    {
        int jhigh = min(i, c.jmax);

        // Forward iteration step, compute Qs in the next time step
        int j = threadIdx.x - c.jmax - scannedWidthIdx;

        real Q = 0;
        if (i <= c.n && j >= -jhigh && j <= jhigh)
        {   
            auto alpha = args.getAlphaAt(optionIdx, i - 1);
            auto expp1 = j == jhigh ? zero : Qs[threadIdx.x + 1] * exp(-(alpha + (j + 1) * c.dr) * c.dt);
            auto expm = Qs[threadIdx.x] * exp(-(alpha + j * c.dr) * c.dt);
            auto expm1 = j == -jhigh ? zero : Qs[threadIdx.x - 1] * exp(-(alpha + (j - 1) * c.dr) * c.dt);

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
                    Q = ((j == -jhigh + 2) ? computeJValue(j - 2, c.jmax, c.M, 1) * Qs[threadIdx.x - 2] * exp(-(alpha + (j - 2) * c.dr) * c.dt) : zero) +
                        computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1 +
                        ((j == jhigh - 2) ? computeJValue(j + 2, c.jmax, c.M, 3) * Qs[threadIdx.x + 2] * exp(-(alpha + (j + 2) * c.dr) * c.dt) : zero);
                }
            }
        }
        __syncthreads();

        Qs[threadIdx.x] = Q > 0 ? Q * exp(-j * c.dr * c.dt) : 0;
        __syncthreads();

        // Repopulate flags
        optionFlags[threadIdx.x] = threadIdx.x == scannedWidthIdx ? c.width : 0;
        __syncthreads();
        
        // Determine the new alpha using equation 30.22
        // by summing up Qs from the next time step
        real Qexp = sgmScanIncBlock<Add<real>>(Qs, optionFlags, threadIdx.x);
        __syncthreads();
        
        if (i <= c.n && threadIdx.x == scannedWidthIdx + c.width - 1)
        {
            real alpha = computeAlpha(Qexp, i-1, c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize);
            args.setAlphaAt(optionIdx, i, alpha);
        }
        Qs[threadIdx.x] = Q;
        __syncthreads();
    }

    // Backward propagation
    Qs[threadIdx.x] = 100; // initialize to 100$

    for (auto i = args.values.maxHeight - 1; i >= 0; --i)
    {
        int jhigh = min(i, c.jmax);

        // Forward iteration step, compute Qs in the next time step
        int j = threadIdx.x - c.jmax - scannedWidthIdx;

        real call = Qs[threadIdx.x];
        __syncthreads();

        if (i < c.n && j >= -jhigh && j <= jhigh)
        {
            auto alpha = args.getAlphaAt(optionIdx, i);
            auto isMaturity = i == ((int)(c.t / c.dt));
            auto callExp = exp(-(alpha + j * c.dr) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * Qs[threadIdx.x] +
                    computeJValue(j, c.jmax, c.M, 2) * Qs[threadIdx.x - 1] +
                    computeJValue(j, c.jmax, c.M, 3) * Qs[threadIdx.x - 2]) *
                        callExp;
            }
            else if (j == -c.jmax)
            {
                // Bottom edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * Qs[threadIdx.x + 2] +
                    computeJValue(j, c.jmax, c.M, 2) * Qs[threadIdx.x + 1] +
                    computeJValue(j, c.jmax, c.M, 3) * Qs[threadIdx.x]) *
                        callExp;
            }
            else
            {
                // Standard branching
                res = (computeJValue(j, c.jmax, c.M, 1) * Qs[threadIdx.x + 1] +
                    computeJValue(j, c.jmax, c.M, 2) * Qs[threadIdx.x] +
                    computeJValue(j, c.jmax, c.M, 3) * Qs[threadIdx.x - 1]) *
                        callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            call = computeCallValue(isMaturity, c, res);
        }
        __syncthreads();

        Qs[threadIdx.x] = call;
        __syncthreads();
    }

    if (c.n > 0 && threadIdx.x == scannedWidthIdx)
    {
        args.values.res[optionIdx] = Qs[scannedWidthIdx + c.jmax];
    }
}

class KernelRunBase
{

private:
    std::chrono::time_point<std::chrono::steady_clock> time_begin;

protected:
    bool isTest;
    int blockSize;
    int maxHeight;
    int maxWidth;

    virtual void runPreprocessing(CudaOptions &cudaOptions, std::vector<real> &results,
        thrust::device_vector<int32_t> &widths, thrust::device_vector<int32_t> &heights) = 0;

    template<class KernelArgsT, class KernelArgsValuesT>
    void runKernel(CudaOptions &cudaOptions, std::vector<real> &results, thrust::device_vector<int32_t> &inds, KernelArgsValuesT &values)
    {
        const int sharedMemorySize = sizeof(real) * blockSize + sizeof(uint16_t) * blockSize;
        const int totalAlphasCount = cudaOptions.N * maxHeight;
        thrust::device_vector<real> alphas(totalAlphasCount);
        thrust::device_vector<real> result(cudaOptions.N);

        if (isTest)
        {
            std::cout << "Running pricing for " << cudaOptions.N << " options with block size " << blockSize << std::endl;
            cudaDeviceSynchronize();
            size_t memoryFree, memoryTotal;
            cudaMemGetInfo(&memoryFree, &memoryTotal);
            std::cout << "Current GPU memory usage " << (memoryTotal - memoryFree) / (1024.0 * 1024.0) << " MB out of " << memoryTotal / (1024.0 * 1024.0) << " MB " << std::endl;
        }

        values.maxHeight = maxHeight;
        values.res = thrust::raw_pointer_cast(result.data());
        values.alphas = thrust::raw_pointer_cast(alphas.data());
        values.inds = thrust::raw_pointer_cast(inds.data());
        KernelArgsT kernelArgs(values);

        auto time_begin_kernel = std::chrono::steady_clock::now();
        kernelMultipleOptionsPerThreadBlock<<<inds.size(), blockSize, sharedMemorySize>>>(cudaOptions, kernelArgs);
        cudaThreadSynchronize();
        auto time_end_kernel = std::chrono::steady_clock::now();
        runtime.KernelRuntime = std::chrono::duration_cast<std::chrono::microseconds>(time_end_kernel - time_begin_kernel).count();

        if (isTest)
        {
            std::cout << "Kernel executed in " << runtime.KernelRuntime << " microsec" << std::endl;
        }

        CudaCheckError();

        // Copy result
        thrust::copy(result.begin(), result.end(), results.begin());

        auto time_end = std::chrono::steady_clock::now();
        runtime.TotalRuntime = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count();

        if (isTest)
        {
            std::cout << "Total execution time " << runtime.TotalRuntime << " microsec" << std::endl;
        }
    }

public:
    CudaRuntime runtime;
    
    void run(const Options &options, const Yield &yield, std::vector<real> &results, 
        const int blockSize = 1024, const SortType sortType = SortType::NONE, bool isTest = false)
    {
        time_begin = std::chrono::steady_clock::now();

        this->isTest = isTest;
        this->blockSize = blockSize;

        thrust::device_vector<real> strikePrices(options.StrikePrices.begin(), options.StrikePrices.end());
        thrust::device_vector<real> maturities(options.Maturities.begin(), options.Maturities.end());
        thrust::device_vector<real> lengths(options.Lengths.begin(), options.Lengths.end());
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

        // Get the max height and width
        maxWidth = thrust::max_element(widths.begin(), widths.end())[0];
        maxHeight = thrust::max_element(heights.begin(), heights.end())[0];

        if (maxWidth > blockSize)
        {
            std::ostringstream oss;
            oss << "Block size (" << blockSize << ") cannot be smaller than max option width (" << maxWidth << ").";
            throw std::invalid_argument(oss.str());
        }

        runPreprocessing(cudaOptions, results, widths, heights);
    }

};

}

}

#endif