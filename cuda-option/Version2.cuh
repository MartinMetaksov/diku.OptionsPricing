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

__device__ void fillArrayColumn(const int count, const real value, real *array, const int totalCount, const int threadId)
{
    auto ptr = getArrayAt(0, array, totalCount, threadId);

    for (auto i = 0; i < count; ++i)
    {
        *ptr = value;
        ptr += totalCount;
    }
}
    
__global__ void
kernelCoalesced(real *res, const OptionConstants *options, real *QsAll, real *QsCopyAll, real *alphasAll, const int totalCount, const int yieldCurveSize)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Out of options check
    if (idx >= totalCount) return;

    auto c = options[idx];
    auto alpha = getYieldAtYear(c.dt, c.termUnit, YieldCurve, yieldCurveSize);
    *getArrayAt(c.jmax, QsAll, totalCount, idx) = one;
    *getArrayAt(0, alphasAll, totalCount, idx) = alpha;

    for (auto i = 1; i <= c.n; ++i)
    {
        const auto jhigh = min(i, c.jmax);
        real alpha_val = 0;

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            
            auto expp1 = j == jhigh ? zero : *getArrayAt(jind + 1, QsAll, totalCount, idx) * exp(-(alpha + computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            auto expm = *getArrayAt(jind, QsAll, totalCount, idx) * exp(-(alpha + computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            auto expm1 = j == -jhigh ? zero : *getArrayAt(jind - 1, QsAll, totalCount, idx)  * exp(-(alpha + computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
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
                    Q = ((j == -jhigh + 2) ? computeJValue(jind - 2, c.dr, c.M, c.width, c.jmax, 1) * *getArrayAt(jind - 2, QsAll, totalCount, idx) * exp(-(alpha + computeJValue(jind - 2, c.dr, c.M, c.width, c.jmax, 0)) * c.dt) : zero) +
                        computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1 +
                        ((j == jhigh - 2) ? computeJValue(jind + 2, c.dr, c.M, c.width, c.jmax, 3) * *getArrayAt(jind + 2, QsAll, totalCount, idx) * exp(-(alpha + computeJValue(jind + 2, c.dr, c.M, c.width, c.jmax, 0)) * c.dt) : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            *getArrayAt(jind, QsCopyAll, totalCount, idx) = Q;
            alpha_val += Q * exp(-computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0) * c.dt);
        }

        alpha = computeAlpha(alpha_val, i-1, c.dt, c.termUnit, YieldCurve, yieldCurveSize);
        *getArrayAt(i, alphasAll, totalCount, idx) = alpha;

        // Switch Qs
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;
        fillArrayColumn(c.width, 0, QsCopyAll, totalCount, idx);
    }
    
    // Backward propagation
    fillArrayColumn(c.width, 100, QsAll, totalCount, idx); // initialize to 100$

    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = *getArrayAt(i, alphasAll, totalCount, idx);
        auto isMaturity = i == ((int)(c.t / c.dt));

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            auto callExp = exp(-(alpha + computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * *getArrayAt(jind, QsAll, totalCount, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * *getArrayAt(jind - 1, QsAll, totalCount, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * *getArrayAt(jind - 2, QsAll, totalCount, idx)) *
                      callExp;
            }
            else if (j == - c.jmax)
            {
                // Bottom edge branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * *getArrayAt(jind + 2, QsAll, totalCount, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * *getArrayAt(jind + 1, QsAll, totalCount, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * *getArrayAt(jind, QsAll, totalCount, idx)) *
                      callExp;
            }
            else
            {
                // Standard branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * *getArrayAt(jind + 1, QsAll, totalCount, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * *getArrayAt(jind, QsAll, totalCount, idx) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * *getArrayAt(jind - 1, QsAll, totalCount, idx)) *
                      callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            *getArrayAt(jind, QsCopyAll, totalCount, idx) = computeCallValue(isMaturity, c, res);
        }

        // Switch call arrays
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;

        fillArrayColumn(c.width, 0, QsCopyAll, totalCount, idx);
    }

    res[idx] = *getArrayAt(c.jmax, QsAll, totalCount, idx);
}

void computeOptionsCoalesced(const vector<OptionConstants> &options, const int yieldSize, vector<real> &results, bool isTest = false)
{
    // Compute padding
    int maxWidth = 0;
    int maxHeight = 0;
    for (auto &option : options)
    {
        if (option.width > maxWidth)
        {
            maxWidth = option.width;
        }
        if (option.n > maxHeight)
        {
            maxHeight = option.n;
        }
    }
    maxHeight += 1; // n + 1 alphas
    
    const auto count = options.size();
    const auto totalQsCount = maxWidth * count;
    const auto totalAlphasCount = maxHeight * count;
    const auto blockSize = 64;
    const auto blockCount = ceil(count / ((float)blockSize));

    if (isTest)
    {
        int memorySize = count * sizeof(real) + count * sizeof(OptionConstants)
                        + 2 * totalQsCount * sizeof(real) + totalAlphasCount * sizeof(real);
        cout << "Running trinomial option pricing for " << count << " options with block size " << blockSize << endl;
        cout << "Global memory size " << memorySize / (1024.0 * 1024.0) << " MB" << endl;
    }

    const auto time_begin = steady_clock::now();

    real *d_result, *d_Qs, *d_QsCopy, *d_alphas;
    OptionConstants *d_options;
    CudaSafeCall(cudaMalloc((void **)&d_result, count * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_options, count * sizeof(OptionConstants)));
    CudaSafeCall(cudaMalloc((void **)&d_Qs, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_QsCopy, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_alphas, totalAlphasCount * sizeof(real)));

    CudaSafeCall(cudaMemcpy(d_options, options.data(), count * sizeof(OptionConstants), cudaMemcpyHostToDevice));

    auto time_begin_kernel = steady_clock::now();
    kernelCoalesced<<<blockCount, blockSize>>>(d_result, d_options, d_Qs, d_QsCopy, d_alphas, count, yieldSize);
    cudaThreadSynchronize();
    auto time_end_kernel = steady_clock::now();

    CudaCheckError();

    // Copy result
    cudaMemcpy(results.data(), d_result, count * sizeof(real), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
    cudaFree(d_options);
    cudaFree(d_Qs);
    cudaFree(d_QsCopy);
    cudaFree(d_alphas);

    auto time_end = steady_clock::now();
    if (isTest)
    {
        cout << "Kernel executed in " << duration_cast<milliseconds>(time_end_kernel - time_begin_kernel).count() << " ms" << endl;
        cout << "Total GPU time: " << duration_cast<milliseconds>(time_end - time_begin).count() << " ms" << endl
            << endl;
    }
}

}

#endif