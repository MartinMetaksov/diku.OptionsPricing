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
    
__global__ void
computeSingleOptionKernel(real *res, OptionConstants *options, real *QsAll, real *QsCopyAll, real *alphasAll, int *QsInd, int *alphasInd, int totalCount)
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
    alphas[0] = getYieldAtYear(c.dt);

    for (auto i = 0; i < c.n; ++i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = alphas[i];

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto qexp = Qs[jind] * exp(-(alpha + computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);

            if (j == jmin)
            {
                // Bottom edge branching
                QsCopy[jind + 2] += computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * qexp; // up two
                QsCopy[jind + 1] += computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * qexp; // up one
                QsCopy[jind] += computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * qexp;     // middle
            }
            else if (j == c.jmax)
            {
                // Top edge branching
                QsCopy[jind] += computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * qexp;     // middle
                QsCopy[jind - 1] += computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * qexp; // down one
                QsCopy[jind - 2] += computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * qexp; // down two
            }
            else
            {
                // Standard branching
                QsCopy[jind + 1] += computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * qexp; // up
                QsCopy[jind] += computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * qexp;     // middle
                QsCopy[jind - 1] += computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * qexp; // down
            }
        }

        // Determine the new alpha using equation 30.22
        // by summing up Qs from the next time step
        auto jhigh1 = min(i + 1, c.jmax);
        real alpha_val = 0;
        for (auto j = -jhigh1; j <= jhigh1; ++j)
        {
            auto jind = j - jmin;      // array index for j
            alpha_val += QsCopy[jind] * exp(-computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0) * c.dt);
        }

        alphas[i + 1] = computeAlpha(alpha_val, i, c.dt);

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
            callCopy[jind] = isMaturity ? max(c.X - res, zero) : res;
        }

        // Switch call arrays
        auto callT = call;
        call = callCopy;
        callCopy = callT;

        thrust::fill_n(thrust::device, callCopy, c.width, 0);
    }

    res[idx] = call[c.jmax];
}

void computeOptions(OptionConstants *options, real *result, int count, bool isTest = false)
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
    auto blockCount = count / (blockSize + 1) + 1;

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

    cudaMemcpy(d_options, options, count * sizeof(OptionConstants), cudaMemcpyHostToDevice);
    cudaMemcpy(d_QsInd, QsInd, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alphasInd, alphasInd, count * sizeof(int), cudaMemcpyHostToDevice);

    auto time_begin_kernel = steady_clock::now();
    computeSingleOptionKernel<<<blockCount, blockSize>>>(d_result, d_options, d_Qs, d_QsCopy, d_alphas, d_QsInd, d_alphasInd, count);
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
