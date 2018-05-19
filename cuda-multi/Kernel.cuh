#ifndef CUDA_KERNEL_MULTI_CUH
#define CUDA_KERNEL_MULTI_CUH

#include "../cuda/CudaDomain.cuh"
#include "../cuda/ScanKernels.cuh"

using namespace chrono;
using namespace trinom;

namespace cuda
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

};

template<class KernelArgsT>
__global__ void kernelMultipleOptionsPerThreadBlock(const CudaOptions options, KernelArgsT args)
{
    extern __shared__ real sh_mem[];
    volatile real *Qs = (real *)&sh_mem;
    volatile real *QCopys = &Qs[blockDim.x];
    volatile int *optionInds = (int *) &QCopys[blockDim.x];
    volatile int *optionFlags = &optionInds[blockDim.x];

    // Compute option indices and init Qs
    optionInds[threadIdx.x] = 0;
    Qs[threadIdx.x] = 0;
    __syncthreads();

    const auto idxBlock = blockIdx.x == 0 ? 0 : args.values.inds[blockIdx.x - 1];
    const auto idx = idxBlock + threadIdx.x;
    const auto nextIdx = args.values.inds[blockIdx.x];
    int width;
    if (idx < nextIdx)    // Don't fetch options from next block
    {
        width = options.Widths[idx];
        optionInds[threadIdx.x] = width;
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
    // Set starting Qs to 1$
    if (idx < nextIdx)
    {
        Qs[scannedWidthIdx + width / 2] = 1;
    }
    optionInds[threadIdx.x] = 0;
    __syncthreads();

    if (idx <= nextIdx)
    {
        optionInds[scannedWidthIdx] = threadIdx.x;
        optionFlags[scannedWidthIdx] = idx == nextIdx ? optionInds[threadIdx.x] : width;
    }
    __syncthreads();

    optionInds[threadIdx.x] = sgmScanIncBlock<Add<int>>(optionInds, optionFlags, threadIdx.x);

    // Get the option and set inital alpha and Q
    OptionConstants c;
    const auto optionIdx = idxBlock + optionInds[threadIdx.x];
    real *alphas = args.values.alphas + args.values.maxHeight * optionIdx;
    if (optionIdx < nextIdx)
    {
        computeConstants(c, options, optionIdx);
        alphas[0] = getYieldAtYear(c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize);
    }
    __syncthreads();

    // TODO: Forward iteration

}

class KernelRunBase
{

protected:
    bool isTest;
    int blockSize;
    int maxHeight;

    virtual void runPreprocessing(CudaOptions &cudaOptions, vector<real> &results,
        thrust::device_vector<int32_t> &widths, thrust::device_vector<int32_t> &heights) = 0;

    template<class KernelArgsT, class KernelArgsValuesT>
    void runKernel(CudaOptions &cudaOptions, vector<real> &results, thrust::device_vector<int32_t> &inds, int32_t sharedMemorySize, KernelArgsValuesT &values)
    {
        const int totalAlphasCount = cudaOptions.N * maxHeight;
        thrust::device_vector<real> alphas(totalAlphasCount);
        thrust::device_vector<real> result(cudaOptions.N);

        if (isTest)
        {
            cout << "Running pricing for " << cudaOptions.N << " options with block size " << blockSize << endl;
            cudaDeviceSynchronize();
            size_t memoryFree, memoryTotal;
            cudaMemGetInfo(&memoryFree, &memoryTotal);
            cout << "Current GPU memory usage " << (memoryTotal - memoryFree) / (1024.0 * 1024.0) << " MB out of " << memoryTotal / (1024.0 * 1024.0) << " MB " << endl;
        }

        values.maxHeight = maxHeight;
        values.res = thrust::raw_pointer_cast(result.data());
        values.alphas = thrust::raw_pointer_cast(alphas.data());
        values.inds = thrust::raw_pointer_cast(inds.data());
        KernelArgsT kernelArgs(values);

        auto time_begin_kernel = steady_clock::now();
        kernelMultipleOptionsPerThreadBlock<<<inds.size(), blockSize, sharedMemorySize>>>(cudaOptions, kernelArgs);
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

public:
    
    void run(const Options &options, const Yield &yield, vector<real> &results, 
        const int blockSize = 1024, const SortType sortType = SortType::NONE, bool isTest = false)
    {
        this->isTest = isTest;
        this->blockSize = blockSize;

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

        // Get the max height
        maxHeight = thrust::max_element(heights.begin(), heights.end())[0];

        runPreprocessing(cudaOptions, results, widths, heights);
    }

};

}

#endif