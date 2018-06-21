#ifndef CUDA_KERNEL_OPTION_CUH
#define CUDA_KERNEL_OPTION_CUH

#include "../cuda/CudaDomain.cuh"

using namespace trinom;

namespace cuda
{

namespace option
{

struct KernelArgsValues
{
    real *res;
    real *QsAll;
    real *QsCopyAll;
    real *alphasAll;
};

/**
Base class for kernel arguments.
Important! Don't call defined pure virtual functions within your implementation.
**/
template<class KernelArgsValuesT>
class KernelArgsBase
{

protected:
    KernelArgsValuesT values;

public:

    KernelArgsBase(KernelArgsValuesT &v) : values(v) { }
    
    __device__ inline int getIdx() const { return threadIdx.x + blockDim.x * blockIdx.x; }

    __device__ virtual void init(const CudaOptions &options) = 0;

    __device__ virtual void switchQs()
    {
        auto QsT = values.QsAll;
        values.QsAll = values.QsCopyAll;
        values.QsCopyAll = QsT;
    }

    __device__ virtual void fillQs(const int count, const int value) = 0;

    __device__ virtual void setQAt(const int index, const real value) = 0;

    __device__ virtual void setQCopyAt(const int index, const real value) = 0;

    __device__ virtual void setAlphaAt(const int index, const real value) = 0;

    __device__ virtual void setResult(const int jmax) = 0;

    __device__ virtual real getQAt(const int index) const = 0;

    __device__ virtual real getAlphaAt(const int index) const = 0;
};

template<class KernelArgsT>
__global__ void kernelOneOptionPerThread(const CudaOptions options, KernelArgsT args)
{
    auto idx = args.getIdx();

    // Out of options check
    if (idx >= options.N) return;

    OptionConstants c;
    computeConstants(c, options, idx);

    args.init(options);
    args.setQAt(c.jmax, one);
    args.setAlphaAt(0, getYieldAtYear(c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize));

    // Forward propagation
    for (auto i = 1; i <= c.n; ++i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = args.getAlphaAt(i-1);
        real alpha_val = 0;

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j            
            
            auto expp1 = j == jhigh ? zero : args.getQAt(jind + 1) * exp(-(alpha + (j + 1) * c.dr) * c.dt);
            auto expm = args.getQAt(jind) * exp(-(alpha + j * c.dr) * c.dt);
            auto expm1 = j == -jhigh ? zero : args.getQAt(jind - 1) * exp(-(alpha + (j - 1) * c.dr) * c.dt);
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
                    Q = ((j == -jhigh + 2) ? computeJValue(j - 2, c.jmax, c.M, 1) * args.getQAt(jind - 2) * exp(-(alpha + (j - 2) * c.dr) * c.dt) : zero) +
                        computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1 +
                        ((j == jhigh - 2) ? computeJValue(j + 2, c.jmax, c.M, 3) * args.getQAt(jind + 2) * exp(-(alpha + (j + 2) * c.dr) * c.dt) : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            args.setQCopyAt(jind, Q); 
            alpha_val += Q * exp(-j * c.dr * c.dt);
        }

        args.setAlphaAt(i, computeAlpha(alpha_val, i-1, c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize));

        // Switch Qs
        args.switchQs();
    }
    
    // Backward propagation
    args.fillQs(c.width, 100); // initialize to 100$

    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = args.getAlphaAt(i);
        auto isMaturity = i == ((int)(c.t / c.dt));

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            auto callExp = exp(-(alpha + j * c.dr) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * args.getQAt(jind) +
                    computeJValue(j, c.jmax, c.M, 2) * args.getQAt(jind - 1) +
                    computeJValue(j, c.jmax, c.M, 3) * args.getQAt(jind - 2)) *
                        callExp;
            }
            else if (j == -c.jmax)
            {
                // Bottom edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * args.getQAt(jind + 2) +
                    computeJValue(j, c.jmax, c.M, 2) * args.getQAt(jind + 1) +
                    computeJValue(j, c.jmax, c.M, 3) * args.getQAt(jind)) *
                        callExp;
            }
            else
            {
                // Standard branching
                res = (computeJValue(j, c.jmax, c.M, 1) * args.getQAt(jind + 1) +
                    computeJValue(j, c.jmax, c.M, 2) * args.getQAt(jind) +
                    computeJValue(j, c.jmax, c.M, 3) * args.getQAt(jind - 1)) *
                        callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            args.setQCopyAt(jind, computeCallValue(isMaturity, c, res));
        }

        // Switch Qs
        args.switchQs();
    }

    args.setResult(c.jmax);
}

class KernelRunBase
{

private:
    std::chrono::time_point<std::chrono::steady_clock> time_begin;

protected:
    bool isTest;
    int blockSize;
    size_t deviceMemory = 0;

    virtual void runPreprocessing(CudaOptions &cudaOptions, std::vector<real> &results,
        thrust::device_vector<int32_t> &widths, thrust::device_vector<int32_t> &heights) = 0;

    template<class KernelArgsT, class KernelArgsValuesT>
    void runKernel(CudaOptions &cudaOptions, std::vector<real> &results, const int totalQsCount, const int totalAlphasCount, KernelArgsValuesT &values)
    {
        thrust::device_vector<real> Qs(totalQsCount);
        thrust::device_vector<real> QsCopy(totalQsCount);
        thrust::device_vector<real> alphas(totalAlphasCount);
        thrust::device_vector<real> result(cudaOptions.N);

        const auto blockCount = ceil(cudaOptions.N / ((float)blockSize));

        if (isTest)
        {
            deviceMemory += vectorsizeof(Qs);
            deviceMemory += vectorsizeof(QsCopy);
            deviceMemory += vectorsizeof(alphas);
            deviceMemory += vectorsizeof(result);

            std::cout << "Running pricing for " << cudaOptions.N << 
            #ifdef USE_DOUBLE
            " double"
            #else
            " float"
            #endif
            << " options with block size " << blockSize << std::endl;
            std::cout << "Qs count " << totalQsCount << ", alphas count " << totalAlphasCount << std::endl;
            std::cout << "Global memory size " << deviceMemory / (1024.0 * 1024.0) << " MB" << std::endl;

            cudaDeviceSynchronize();
            size_t memoryFree, memoryTotal;
            cudaMemGetInfo(&memoryFree, &memoryTotal);
            std::cout << "Current GPU memory usage " << (memoryTotal - memoryFree) / (1024.0 * 1024.0) << " MB out of " << memoryTotal / (1024.0 * 1024.0) << " MB " << std::endl;
        }

        values.res = thrust::raw_pointer_cast(result.data());
        values.QsAll = thrust::raw_pointer_cast(Qs.data());
        values.QsCopyAll = thrust::raw_pointer_cast(QsCopy.data());
        values.alphasAll = thrust::raw_pointer_cast(alphas.data());
        KernelArgsT kernelArgs(values);

        auto time_begin_kernel = std::chrono::steady_clock::now();
        kernelOneOptionPerThread<<<blockCount, blockSize>>>(cudaOptions, kernelArgs);
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
        const int blockSize = 64, const SortType sortType = SortType::NONE, bool isTest = false)
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

        if (isTest)
        {
            deviceMemory += vectorsizeof(strikePrices);
            deviceMemory += vectorsizeof(maturities);
            deviceMemory += vectorsizeof(lengths);
            deviceMemory += vectorsizeof(termUnits);
            deviceMemory += vectorsizeof(termStepCounts);
            deviceMemory += vectorsizeof(reversionRates);
            deviceMemory += vectorsizeof(volatilities);
            deviceMemory += vectorsizeof(types);
            deviceMemory += vectorsizeof(yieldPrices);
            deviceMemory += vectorsizeof(yieldTimeSteps);
            deviceMemory += vectorsizeof(widths);
            deviceMemory += vectorsizeof(heights);
        }

        CudaOptions cudaOptions(options, yield.N, sortType, isTest, strikePrices, maturities, lengths, termUnits, 
            termStepCounts, reversionRates, volatilities, types, yieldPrices, yieldTimeSteps, widths, heights);

        runPreprocessing(cudaOptions, results, widths, heights);
    }

};

}

}

#endif