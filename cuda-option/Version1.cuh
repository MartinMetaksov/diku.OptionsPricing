#ifndef CUDA_VERSION_1_CUH
#define CUDA_VERSION_1_CUH

#include "../cuda/CudaDomain.cuh"

using namespace chrono;
using namespace trinom;

namespace cuda
{

__device__ inline void fill_n(real *array, const int count, const int value)
{
    for (auto i = 0; i < count; ++i)
    {
        array[i] = value;
    }
}

__global__ void
kernelNaive(const CudaOptions options, real *res, real *QsAll, real *QsCopyAll, real *alphasAll)
{
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Out of options check
    if (idx >= options.N) return;

    OptionConstants c;
    computeConstants(c, options, idx);

    auto QsInd = idx == 0 ? 0 : options.Widths[idx - 1];
    auto alphasInd = idx == 0 ? 0 : options.Heights[idx - 1];
    auto Qs = QsAll + QsInd;
    auto QsCopy = QsCopyAll + QsInd;
    auto alphas = alphasAll + alphasInd;
    Qs[c.jmax] = one;
    alphas[0] = getYieldAtYear(c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize);

    for (auto i = 1; i <= c.n; ++i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = alphas[i-1];
        real alpha_val = 0;

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j            
            
            auto expp1 = j == jhigh ? zero : Qs[jind + 1] * exp(-(alpha + (j + 1) * c.dr) * c.dt);
            auto expm = Qs[jind] * exp(-(alpha + j * c.dr) * c.dt);
            auto expm1 = j == -jhigh ? zero : Qs[jind - 1] * exp(-(alpha + (j - 1) * c.dr) * c.dt);
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
                    Q = ((j == -jhigh + 2) ? computeJValue(j - 2, c.jmax, c.M, 1) * Qs[jind - 2] * exp(-(alpha + (j - 2) * c.dr) * c.dt) : zero) +
                        computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1 +
                        ((j == jhigh - 2) ? computeJValue(j + 2, c.jmax, c.M, 3) * Qs[jind + 2] * exp(-(alpha + (j + 2) * c.dr) * c.dt) : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            QsCopy[jind] = Q; 
            alpha_val += Q * exp(-j * c.dr * c.dt);
        }

        alphas[i] = computeAlpha(alpha_val, i-1, c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize);

        // Switch Qs
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;
        
        fill_n(QsCopy, c.width, 0);
    }
    
    // Backward propagation
    auto call = Qs; // call[j]
    auto callCopy = QsCopy;

    fill_n(call, c.width, 100); // initialize to 100$

    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = alphas[i];
        auto isMaturity = i == ((int)(c.t / c.dt));

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            auto callExp = exp(-(alpha + j * c.dr) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * call[jind] +
                    computeJValue(j, c.jmax, c.M, 2) * call[jind - 1] +
                    computeJValue(j, c.jmax, c.M, 3) * call[jind - 2]) *
                        callExp;
            }
            else if (j == -c.jmax)
            {
                // Bottom edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * call[jind + 2] +
                    computeJValue(j, c.jmax, c.M, 2) * call[jind + 1] +
                    computeJValue(j, c.jmax, c.M, 3) * call[jind]) *
                        callExp;
            }
            else
            {
                // Standard branching
                res = (computeJValue(j, c.jmax, c.M, 1) * call[jind + 1] +
                    computeJValue(j, c.jmax, c.M, 2) * call[jind] +
                    computeJValue(j, c.jmax, c.M, 3) * call[jind - 1]) *
                        callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            callCopy[jind] = computeCallValue(isMaturity, c, res);
        }

        // Switch call arrays
        auto callT = call;
        call = callCopy;
        callCopy = callT;

        fill_n(callCopy, c.width, 0);
    }

    res[idx] = call[c.jmax];
}

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

    auto time_begin_kernel = steady_clock::now();
    kernelNaive<<<blockCount, blockSize>>>(cudaOptions, d_result, d_Qs, d_QsCopy, d_alphas);
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