#ifndef CUDA_OPTION_CUH
#define CUDA_OPTION_CUH

#include "../common/Real.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/Domain.hpp"
#include "../cuda/CudaErrors.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

using namespace chrono;
using namespace trinom;

namespace cuda
{

__constant__ Yield YieldCurve[100];
    
__global__ void
computeSingleOptionKernel(real *res, OptionConstants *options, real *QsAll, real *QsCopyAll, real *alphasAll, 
    int *QsInd, int *alphasInd, int totalCount, int yieldCurveSize)
{
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Out of options check
    if (idx >= totalCount) return;

    auto c = options[idx];
    auto Qs = QsAll + QsInd[idx];
    auto QsCopy = QsCopyAll + QsInd[idx];
    auto alphas = alphasAll + alphasInd[idx];
    auto jmin = -c.jmax;
    Qs[c.jmax] = one;
    alphas[0] = getYieldAtYear(c.dt, c.termUnit, YieldCurve, yieldCurveSize);

    for (auto i = 1; i <= c.n; ++i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = alphas[i-1];
        real alpha_val = 0;

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j

            auto expp1 = j == jhigh ? zero : Qs[jind + 1] * exp(-(alpha + computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            auto expm = Qs[jind] * exp(-(alpha + computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            auto expm1 = j == -jhigh ? zero : Qs[jind - 1] * exp(-(alpha + computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            
            if (i == 1) {
                if (j == -jhigh) {
                    QsCopy[jind] = 
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == jhigh) {
                    QsCopy[jind] = 
                        computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1;
                } else {
                    QsCopy[jind] = 
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm;
                }
            }
            else if (i <= c.jmax) {
                if (j == -jhigh) {
                    QsCopy[jind] = 
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == -jhigh + 1) {
                    QsCopy[jind] = 
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == jhigh) {
                    QsCopy[jind] =
                        computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1;
                } else if (j == jhigh - 1) {
                    QsCopy[jind] =
                        computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm;
                } else {
                    QsCopy[jind] =
                        computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                }
            } else {
                if (j == -jhigh) {
                    QsCopy[jind] =
                            computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * expm +
                            computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == -jhigh + 1) {
                    QsCopy[jind] =
                            computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 2) * expm1 +
                            computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                            computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                            
                } else if (j == jhigh) {
                    QsCopy[jind] =
                            computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                            computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * expm;
                } else if (j == jhigh - 1) {
                    QsCopy[jind] =
                            computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                            computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                            computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 2) * expp1;
                            
                } else {
                    QsCopy[jind] =
                            ((j == -jhigh + 2) ? computeJValue(jind - 2, c.dr, c.M, c.width, c.jmax, 1) * Qs[jind - 2] * exp(-(alpha + computeJValue(jind - 2, c.dr, c.M, c.width, c.jmax, 0)) * c.dt) : zero) +
                            computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                            computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                            computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1 +
                            ((j == jhigh - 2) ? computeJValue(jind + 2, c.dr, c.M, c.width, c.jmax, 3) * Qs[jind + 2] * exp(-(alpha + computeJValue(jind + 2, c.dr, c.M, c.width, c.jmax, 0)) * c.dt) : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            alpha_val += QsCopy[jind] * exp(-computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0) * c.dt);
        }

        alphas[i] = computeAlpha(alpha_val, i-1, c.dt, c.termUnit, YieldCurve, yieldCurveSize);

        // Switch Qs
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;
        thrust::fill_n(thrust::device, QsCopy, c.width, 0);
    }
    
    // Backward propagation
    auto call = Qs; // call[j]
    auto callCopy = QsCopy;

    thrust::fill_n(thrust::device, call, c.width, 100); // initialize to 100$

    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = alphas[i];
        auto isMaturity = i == ((int)(c.t / c.dt));

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j

            auto callExp = exp(-(alpha + computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * call[jind] +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * call[jind - 1] +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * call[jind - 2]) *
                      callExp;
            }
            else if (j == jmin)
            {
                // Bottom edge branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * call[jind + 2] +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * call[jind + 1] +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * call[jind]) *
                      callExp;
            }
            else
            {
                // Standard branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * call[jind + 1] +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * call[jind] +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * call[jind - 1]) *
                      callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            callCopy[jind] = computeCallValue(isMaturity, c, res);
        }

        // Switch call arrays
        auto callT = call;
        call = callCopy;
        callCopy = callT;

        thrust::fill_n(thrust::device, callCopy, c.width, 0);
    }

    res[idx] = call[c.jmax];
}

void computeOptions(OptionConstants *options, real *result, int count, const vector<Yield> &yield, bool isTest = false)
{
    // Compute indices
    int* QsInd = new int[count];
    int* alphasInd = new int[count];
    QsInd[0] = 0;
    alphasInd[0] = 0;
    int totalQsCount = 0;
    int totalAlphasCount = 0;
    for (auto i = 0; i < count - 1; ++i)
    {
        auto &option = options[i];
        totalQsCount += option.width;
        totalAlphasCount += option.n + 1;
        QsInd[i + 1] = totalQsCount;
        alphasInd[i + 1] = totalAlphasCount;
    }
    totalQsCount += options[count - 1].width;
    totalAlphasCount += options[count - 1].n + 1;
    
    auto blockSize = 32;
    auto blockCount = ceil(count / ((float)blockSize));

    if (isTest)
    {
        int memorySize = count * sizeof(real) + count * sizeof(OptionConstants) + 2 * count * sizeof(int)
                        + 2 * totalQsCount * sizeof(real) + totalAlphasCount * sizeof(real);
        cout << "Running trinomial option pricing for " << count << " options with block size " << blockSize << endl;
        cout << "Global memory size " << memorySize / (1024.0 * 1024.0) << " MB" << endl;
    }

    auto time_begin = steady_clock::now();

    real *d_result, *d_Qs, *d_QsCopy, *d_alphas;
    int *d_QsInd, *d_alphasInd;
    OptionConstants *d_options;
    CudaSafeCall(cudaMalloc((void **)&d_result, count * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_options, count * sizeof(OptionConstants)));
    CudaSafeCall(cudaMalloc((void **)&d_QsInd, count * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_alphasInd, count * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_Qs, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_QsCopy, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_alphas, totalAlphasCount * sizeof(real)));

    CudaSafeCall(cudaMemcpyToSymbol(YieldCurve, yield.data(), yield.size() * sizeof(Yield)));
    CudaSafeCall(cudaMemcpy(d_options, options, count * sizeof(OptionConstants), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_QsInd, QsInd, count * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_alphasInd, alphasInd, count * sizeof(int), cudaMemcpyHostToDevice));

    auto time_begin_kernel = steady_clock::now();
    computeSingleOptionKernel<<<blockCount, blockSize>>>(d_result, d_options, d_Qs, d_QsCopy, d_alphas, d_QsInd, d_alphasInd, count, yield.size());
    cudaThreadSynchronize();
    auto time_end_kernel = steady_clock::now();

    CudaCheckError();

    // Copy result
    cudaMemcpy(result, d_result, count * sizeof(real), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
    cudaFree(d_options);
    cudaFree(d_QsInd);
    cudaFree(d_alphasInd);
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
