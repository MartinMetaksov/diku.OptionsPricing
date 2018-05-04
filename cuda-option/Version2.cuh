#ifndef CUDA_VERSION_2_CUH
#define CUDA_VERSION_2_CUH

#include "../cuda/CudaDomain.cuh"

using namespace chrono;
using namespace trinom;

namespace cuda
{

__device__ inline real* getArrayAt(const int index, real *array, const int count, const int threadId)
{
    return array + index * count + threadId;
}

__device__ inline void fillArrayColumn(const int count, const real value, real *array, const int totalCount, const int threadId)
{
    auto ptr = getArrayAt(0, array, totalCount, threadId);

    for (auto i = 0; i < count; ++i)
    {
        *ptr = value;
        ptr += totalCount;
    }
}
    
__global__ void
kernelCoalesced(const CudaOptions options, real *res, real *QsAll, real *QsCopyAll, real *alphasAll)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Out of options check
    if (idx >= options.N) return;

    OptionConstants c;
    computeConstants(c, options, idx);

    auto alpha = getYieldAtYear(c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize);
    *getArrayAt(c.jmax, QsAll, options.N, idx) = one;
    *getArrayAt(0, alphasAll, options.N, idx) = alpha;

    for (auto i = 1; i <= c.n; ++i)
    {
        const auto jhigh = min(i, c.jmax);
        real alpha_val = 0;

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            
            auto expp1 = j == jhigh ? zero : *getArrayAt(jind + 1, QsAll, options.N, idx) * exp(-(alpha + computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            auto expm = *getArrayAt(jind, QsAll, options.N, idx) * exp(-(alpha + computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            auto expm1 = j == -jhigh ? zero : *getArrayAt(jind - 1, QsAll, options.N, idx)  * exp(-(alpha + computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            real Q;

            if (i == 1) {
                if (j == -jhigh) {
                    Q = computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == jhigh) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1;
                } else {
                    Q = computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm;
                }
            }
            else if (i <= c.jmax) {
                if (j == -jhigh) {
                    Q = computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == -jhigh + 1) {
                    Q = computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == jhigh) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1;
                } else if (j == jhigh - 1) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm;
                } else {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                }
            } else {
                if (j == -jhigh) {
                    Q = computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == -jhigh + 1) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 2) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                            
                } else if (j == jhigh) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * expm;
                } else if (j == jhigh - 1) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 2) * expp1;
                            
                } else {
                    Q = ((j == -jhigh + 2) ? computeJValue(jind - 2, c.dr, c.M, c.width, c.jmax, 1) * *getArrayAt(jind - 2, QsAll, options.N, idx) * exp(-(alpha + computeJValue(jind - 2, c.dr, c.M, c.width, c.jmax, 0)) * c.dt) : zero) +
                        computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1 +
                        ((j == jhigh - 2) ? computeJValue(jind + 2, c.dr, c.M, c.width, c.jmax, 3) * *getArrayAt(jind + 2, QsAll, options.N, idx) * exp(-(alpha + computeJValue(jind + 2, c.dr, c.M, c.width, c.jmax, 0)) * c.dt) : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            *getArrayAt(jind, QsCopyAll, options.N, idx) = Q;
            alpha_val += Q * exp(-computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0) * c.dt);
        }

        alpha = computeAlpha(alpha_val, i-1, c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize);
        *getArrayAt(i, alphasAll, options.N, idx) = alpha;

        // Switch Qs
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;
        fillArrayColumn(c.width, 0, QsCopyAll, options.N, idx);
    }
    
    // Backward propagation
    fillArrayColumn(c.width, 100, QsAll, options.N, idx); // initialize to 100$

    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = *getArrayAt(i, alphasAll, options.N, idx);
        auto isMaturity = i == ((int)(c.t / c.dt));

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            auto callExp = exp(-(alpha + computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * *getArrayAt(jind, QsAll, options.N, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * *getArrayAt(jind - 1, QsAll, options.N, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * *getArrayAt(jind - 2, QsAll, options.N, idx)) *
                      callExp;
            }
            else if (j == - c.jmax)
            {
                // Bottom edge branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * *getArrayAt(jind + 2, QsAll, options.N, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * *getArrayAt(jind + 1, QsAll, options.N, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * *getArrayAt(jind, QsAll, options.N, idx)) *
                      callExp;
            }
            else
            {
                // Standard branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * *getArrayAt(jind + 1, QsAll, options.N, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * *getArrayAt(jind, QsAll, options.N, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * *getArrayAt(jind - 1, QsAll, options.N, idx)) *
                      callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            *getArrayAt(jind, QsCopyAll, options.N, idx) = computeCallValue(isMaturity, c, res);
        }

        // Switch call arrays
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;

        fillArrayColumn(c.width, 0, QsCopyAll, options.N, idx);
    }

    res[idx] = *getArrayAt(c.jmax, QsAll, options.N, idx);
}

void computeOptionsCoalesced(const Options &options, const Yield &yield, vector<real> &results, const int blockSize, bool isTest = false)
{
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

    CudaOptions cudaOptions(options, yield.N, strikePrices, maturities, lengths, termUnits, 
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
        int memorySize = options.N * sizeof(real) + options.N * sizeof(OptionConstants)
                        + 2 * totalQsCount * sizeof(real) + totalAlphasCount * sizeof(real);
        cout << "Running trinomial option pricing for " << options.N << " options with block size " << blockSize << endl;
        cout << "Global memory size " << memorySize / (1024.0 * 1024.0) << " MB" << endl;
    }

    const auto time_begin = steady_clock::now();

    auto d_result = thrust::raw_pointer_cast(result.data());
    auto d_Qs = thrust::raw_pointer_cast(Qs.data());
    auto d_QsCopy = thrust::raw_pointer_cast(QsCopy.data());
    auto d_alphas = thrust::raw_pointer_cast(alphas.data());

    auto time_begin_kernel = steady_clock::now();
    kernelCoalesced<<<blockCount, blockSize>>>(cudaOptions, d_result, d_Qs, d_QsCopy, d_alphas);
    cudaThreadSynchronize();
    auto time_end_kernel = steady_clock::now();
    if (isTest)
    {
        cout << "Kernel executed in " << duration_cast<milliseconds>(time_end_kernel - time_begin_kernel).count() << " ms" << endl;
    }

    CudaCheckError();

    // Copy result
    thrust::copy(result.begin(), result.end(), results.begin());
}

}

#endif