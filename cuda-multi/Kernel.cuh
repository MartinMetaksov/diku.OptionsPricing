#ifndef CUDA_KERNEL_MULTI_CUH
#define CUDA_KERNEL_MULTI_CUH

#include <stdexcept>
#include <sstream>

#include "../cuda/CudaDomain.cuh"
#include "../cuda/ScanKernels.cuh"

using namespace trinom;

namespace cuda
{

namespace multi
{

volatile extern __shared__ char sh_mem[];

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

    __device__ virtual void init(const int optionIdxBlock, const int idxBlock, const int idxBlockNext, const int optionCount) = 0;

    __device__ virtual void setAlphaAt(const int index, const real value) = 0;

    __device__ virtual real getAlphaAt(const int index) const = 0;

    __device__ virtual int getMaxHeight() const = 0;

    __device__ virtual int getOptionIdx() const = 0;

    __device__ inline volatile real* getQs()
    {
        return (real *)&sh_mem;
    }

    __device__ inline volatile int32_t* getOptionInds()
    {
        return (int32_t *)&sh_mem;  // Sharing the same array with Qs!
    }

    __device__ inline volatile uint16_t* getOptionFlags()
    {
        return (uint16_t *)&sh_mem[blockDim.x * sizeof(real)];
    }
};

template<class KernelArgsT>
__global__ void kernelMultipleOptionsPerThreadBlock(const KernelOptions options, KernelArgsT args)
{
    // Compute option indices and init Qs
    const auto idxBlock = blockIdx.x == 0 ? 0 : args.values.inds[blockIdx.x - 1];
    const auto idxBlockNext = args.values.inds[blockIdx.x];
    const auto idx = idxBlock + threadIdx.x;
    int32_t width; 
    if (idx < idxBlockNext)    // Don't fetch options from next block
    {
        width = options.Widths[idx];
        args.getOptionInds()[threadIdx.x] = width;
    }
    else
    {
        args.getOptionInds()[threadIdx.x] = 0;
    }
    args.getOptionFlags()[threadIdx.x] = threadIdx.x == 0 ? blockDim.x : 0;
    __syncthreads();

    // Scan widths
    // TODO: maybe use scanIncBlock<Add<int32_t>>(args.getOptionInds());
    sgmScanIncBlock<Add<int32_t>>(args.getOptionInds(), args.getOptionFlags());
    
    int scannedWidthIdx = -1;
    if (idx <= idxBlockNext)
    {
        scannedWidthIdx = threadIdx.x == 0 ? 0 : args.getOptionInds()[threadIdx.x - 1];
    }
    __syncthreads();

    // Send option indices to all threads
    args.getOptionInds()[threadIdx.x] = 0;
    args.getOptionFlags()[threadIdx.x] = 0;
    __syncthreads();

    if (idx < idxBlockNext)
    {
        args.getOptionInds()[scannedWidthIdx] = threadIdx.x;
        args.getOptionFlags()[scannedWidthIdx] = width;
    }
    else if (idx == idxBlockNext && scannedWidthIdx < blockDim.x) // fake option to fill block
    {
        args.getOptionInds()[scannedWidthIdx] = threadIdx.x;
        args.getOptionFlags()[scannedWidthIdx] = blockDim.x - scannedWidthIdx;
    }
    __syncthreads();

    sgmScanIncBlock<Add<int32_t>>(args.getOptionInds(), args.getOptionFlags());

    // Let all threads know about their Q start
    if (idx <= idxBlockNext)
    {
        args.getOptionFlags()[threadIdx.x] = scannedWidthIdx;
    }
    __syncthreads();
    scannedWidthIdx = args.getOptionFlags()[args.getOptionInds()[threadIdx.x]];

    // Get the option and compute its constants
    OptionConstants c;
    args.init(args.getOptionInds()[threadIdx.x], idxBlock, idxBlockNext, options.N);
    if (args.getOptionIdx() < idxBlockNext)
    {
        computeConstants(c, options, args.getOptionIdx());
    }
    else
    {
        c.n = 0;
        c.width = blockDim.x - scannedWidthIdx;
    }
    __syncthreads();

    // Zero out Qs
    args.getQs()[threadIdx.x] = 0;
    __syncthreads();

    // Set the initial alpha and Q values
    if (threadIdx.x == scannedWidthIdx && args.getOptionIdx() < idxBlockNext)
    {
        args.setAlphaAt(0, getYieldAtYear(c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize));
        args.getQs()[scannedWidthIdx + c.jmax] = 1;    // Set starting Qs to 1$
    }
    __syncthreads();

    // Forward propagation
    for (int i = 1; i <= args.getMaxHeight(); ++i)
    {
        int jhigh = min(i, c.jmax);

        // Forward iteration step, compute Qs in the next time step
        int j = threadIdx.x - c.jmax - scannedWidthIdx;

        real Q = 0;
        if (i <= c.n && j >= -jhigh && j <= jhigh)
        {   
            auto alpha = args.getAlphaAt(i - 1);
            auto expp1 = j == jhigh ? zero : args.getQs()[threadIdx.x + 1] * exp(-(alpha + (j + 1) * c.dr) * c.dt);
            auto expm = args.getQs()[threadIdx.x] * exp(-(alpha + j * c.dr) * c.dt);
            auto expm1 = j == -jhigh ? zero : args.getQs()[threadIdx.x - 1] * exp(-(alpha + (j - 1) * c.dr) * c.dt);

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
                    Q = ((j == -jhigh + 2) ? computeJValue(j - 2, c.jmax, c.M, 1) * args.getQs()[threadIdx.x - 2] * exp(-(alpha + (j - 2) * c.dr) * c.dt) : zero) +
                        computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1 +
                        ((j == jhigh - 2) ? computeJValue(j + 2, c.jmax, c.M, 3) * args.getQs()[threadIdx.x + 2] * exp(-(alpha + (j + 2) * c.dr) * c.dt) : zero);
                }
            }
        }
        __syncthreads();

        args.getQs()[threadIdx.x] = Q > zero ? Q * exp(-j * c.dr * c.dt) : zero;
        __syncthreads();

        // Repopulate flags
        args.getOptionFlags()[threadIdx.x] = threadIdx.x == scannedWidthIdx ? c.width : 0;
        __syncthreads();
        
        // Determine the new alpha using equation 30.22
        // by summing up Qs from the next time step
        real Qexp = sgmScanIncBlock<Add<real>>(args.getQs(), args.getOptionFlags());
        
        if (i <= c.n && threadIdx.x == scannedWidthIdx + c.width - 1)
        {
            real alpha = computeAlpha(Qexp, i-1, c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize);
            args.setAlphaAt(i, alpha);
        }
        args.getQs()[threadIdx.x] = Q;
        __syncthreads();
    }

    // Backward propagation
    args.getQs()[threadIdx.x] = 100; // initialize to 100$

    for (auto i = args.getMaxHeight() - 1; i >= 0; --i)
    {
        int jhigh = min(i, c.jmax);

        // Forward iteration step, compute Qs in the next time step
        int j = threadIdx.x - c.jmax - scannedWidthIdx;

        real call = args.getQs()[threadIdx.x];

        if (i < c.n && j >= -jhigh && j <= jhigh)
        {
            auto alpha = args.getAlphaAt(i);
            auto isMaturity = i == ((int)(c.t / c.dt));
            auto callExp = exp(-(alpha + j * c.dr) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * args.getQs()[threadIdx.x] +
                    computeJValue(j, c.jmax, c.M, 2) * args.getQs()[threadIdx.x - 1] +
                    computeJValue(j, c.jmax, c.M, 3) * args.getQs()[threadIdx.x - 2]) *
                        callExp;
            }
            else if (j == -c.jmax)
            {
                // Bottom edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * args.getQs()[threadIdx.x + 2] +
                    computeJValue(j, c.jmax, c.M, 2) * args.getQs()[threadIdx.x + 1] +
                    computeJValue(j, c.jmax, c.M, 3) * args.getQs()[threadIdx.x]) *
                        callExp;
            }
            else
            {
                // Standard branching
                res = (computeJValue(j, c.jmax, c.M, 1) * args.getQs()[threadIdx.x + 1] +
                    computeJValue(j, c.jmax, c.M, 2) * args.getQs()[threadIdx.x] +
                    computeJValue(j, c.jmax, c.M, 3) * args.getQs()[threadIdx.x - 1]) *
                        callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            call = computeCallValue(isMaturity, c.X, c.type, res);
        }
        __syncthreads();

        args.getQs()[threadIdx.x] = call;
        __syncthreads();
    }
    
    if (args.getOptionIdx() < idxBlockNext && threadIdx.x == scannedWidthIdx)
    {
        args.values.res[args.getOptionIdx()] = args.getQs()[scannedWidthIdx + c.jmax];
    }
}

class KernelRunBase
{

private:
    std::chrono::time_point<std::chrono::steady_clock> TimeBegin;
    bool IsTest;

protected:
    int BlockSize;

    virtual void runPreprocessing(CudaOptions &options, std::vector<real> &results) = 0;

    template<class KernelArgsT, class KernelArgsValuesT>
    void runKernel(CudaOptions &options, std::vector<real> &results, thrust::device_vector<int32_t> &inds, KernelArgsValuesT &values, const int totalAlphasCount)
    {
        const int sharedMemorySize = (sizeof(real) + sizeof(uint16_t)) * BlockSize;
        thrust::device_vector<real> alphas(totalAlphasCount);
        thrust::device_vector<real> result(options.N);

        options.DeviceMemory += vectorsizeof(inds);
        options.DeviceMemory += vectorsizeof(alphas);
        options.DeviceMemory += vectorsizeof(result);
        runtime.DeviceMemory = options.DeviceMemory;

        if (IsTest)
        {
            std::cout << "Running pricing for " << options.N << 
            #ifdef USE_DOUBLE
            " double"
            #else
            " float"
            #endif
            << " options with block size " << BlockSize << std::endl;
            std::cout << "Shared memory size " << sharedMemorySize << ", alphas count " << totalAlphasCount << std::endl;
            std::cout << "Global memory size " << options.DeviceMemory / (1024.0 * 1024.0) << " MB" << std::endl;

            cudaDeviceSynchronize();
            size_t memoryFree, memoryTotal;
            cudaMemGetInfo(&memoryFree, &memoryTotal);
            std::cout << "Current GPU memory usage " << (memoryTotal - memoryFree) / (1024.0 * 1024.0) << " MB out of " << memoryTotal / (1024.0 * 1024.0) << " MB " << std::endl;
        }

        values.res = thrust::raw_pointer_cast(result.data());
        values.alphas = thrust::raw_pointer_cast(alphas.data());
        values.inds = thrust::raw_pointer_cast(inds.data());
        KernelArgsT kernelArgs(values);

        auto time_begin_kernel = std::chrono::steady_clock::now();
        kernelMultipleOptionsPerThreadBlock<<<inds.size(), BlockSize, sharedMemorySize>>>(options.KernelOptions, kernelArgs);
        cudaDeviceSynchronize();
        auto time_end_kernel = std::chrono::steady_clock::now();
        runtime.KernelRuntime = std::chrono::duration_cast<std::chrono::microseconds>(time_end_kernel - time_begin_kernel).count();

        CudaCheckError();

        if (IsTest)
        {
            std::cout << "Kernel executed in " << runtime.KernelRuntime << " microsec" << std::endl;
        }

        // Sort result
        options.sortResult(result);

        // Stop timing
        auto timeEnd = std::chrono::steady_clock::now();
        runtime.TotalRuntime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - TimeBegin).count();

        if (IsTest)
        {
            std::cout << "Total execution time " << runtime.TotalRuntime << " microsec" << std::endl;
        }

        // Copy result to host
        thrust::copy(result.begin(), result.end(), results.begin());
        cudaDeviceSynchronize();
    }

public:
    CudaRuntime runtime;
    
    void run(const Options &options, const Yield &yield, std::vector<real> &results, 
        const int blockSize = -1, const SortType sortType = SortType::NONE, bool isTest = false)
    {
        CudaOptions cudaOptions(options, yield);

        // Start timing when input is copied to device
        cudaDeviceSynchronize();
        auto timeBegin = std::chrono::steady_clock::now();

        cudaOptions.initialize();

        // Get the max width
        auto maxWidth = *(thrust::max_element(cudaOptions.Widths.begin(), cudaOptions.Widths.end()));

        BlockSize = blockSize;
        if (BlockSize <= 0) 
        {
            // Compute the smallest block size for the max width
            BlockSize = ((maxWidth + 32 - 1) / 32) * 32;
        }

        if (maxWidth > BlockSize)
        {
            std::ostringstream oss;
            oss << "Block size (" << BlockSize << ") cannot be smaller than max option width (" << maxWidth << ").";
            throw std::invalid_argument(oss.str());
        }

        run(cudaOptions, results, timeBegin, blockSize, sortType, isTest);
    }

    void run(CudaOptions &cudaOptions, std::vector<real> &results, const std::chrono::time_point<std::chrono::steady_clock> timeBegin, 
        const int blockSize, const SortType sortType = SortType::NONE, const bool isTest = false)
    {
        TimeBegin = timeBegin;
        IsTest = isTest;

        cudaOptions.sortOptions(sortType, isTest);
        runPreprocessing(cudaOptions, results);
    }

};

}

}

#endif