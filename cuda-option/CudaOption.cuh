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

__device__ inline real* getArrayAt(const int index, real *array, const int count, const int threadId)
{
    return array + index * count + threadId;
}

__device__ inline int getNumUnpaddedFields(const int blockSize, const int blockId, int *maxs) 
{
    // todo: function is incorrect ... 
    int res = 0;
    for(int i = 0; i < blockId; ++i) {
        if (maxs[i] < maxs[i+1]) {
            res += blockSize * (maxs[blockId] - maxs[i]);
        }
    }
    return res;
}

__device__ inline real* getUnpaddedArrayAt(const int index, real *array, const int count, const int threadId, const int blockId, int blockSize, int *maxs)
{
    return array + index * count + threadId - getNumUnpaddedFields(blockSize, blockId, maxs);
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

/*
 * Forward helper using gather
 */
__device__ inline real forward_helper(
    const int idx,
    const int i, 
    const int j, 
    const int jind,
    const int jhigh, 
    const real dr, 
    const real dt, 
    const real M, 
    const int width, 
    const int jmax,
    real* QsAll,
    const int totalCount,
    const real alpha,
    bool isCoalesced) 
{
    auto expp1 = isCoalesced ? 
        (j == jhigh ? zero : *getArrayAt(jind + 1, QsAll, totalCount, idx) * exp(-(alpha + computeJValue(jind + 1, dr, M, width, jmax, 0)) * dt)) : 
        (j == jhigh ? zero : QsAll[jind + 1] * exp(-(alpha + computeJValue(jind + 1, dr, M, width, jmax, 0)) * dt));
    auto expm = isCoalesced ? 
        (*getArrayAt(jind, QsAll, totalCount, idx) * exp(-(alpha + computeJValue(jind, dr, M, width, jmax, 0)) * dt)) : 
        (QsAll[jind] * exp(-(alpha + computeJValue(jind, dr, M, width, jmax, 0)) * dt));
    auto expm1 = isCoalesced ? 
        (j == -jhigh ? zero : *getArrayAt(jind - 1, QsAll, totalCount, idx)  * exp(-(alpha + computeJValue(jind - 1, dr, M, width, jmax, 0)) * dt)) : 
        (j == -jhigh ? zero : QsAll[jind - 1] * exp(-(alpha + computeJValue(jind - 1, dr, M, width, jmax, 0)) * dt));

    if (i == 1) {
        if (j == -jhigh) {
            return computeJValue(jind + 1, dr, M, width, jmax, 3) * expp1;
        } else if (j == jhigh) {
            return computeJValue(jind - 1, dr, M, width, jmax, 1) * expm1;
        } else {
            return computeJValue(jind, dr, M, width, jmax, 2) * expm;
        }
    }

    else if (i <= jmax) {
        if (j == -jhigh) {
            return computeJValue(jind + 1, dr, M, width, jmax, 3) * expp1;
        } else if (j == -jhigh + 1) {
            return computeJValue(jind, dr, M, width, jmax, 2) * expm +
                computeJValue(jind + 1, dr, M, width, jmax, 3) * expp1;
        } else if (j == jhigh) {
            return computeJValue(jind - 1, dr, M, width, jmax, 1) * expm1;
        } else if (j == jhigh - 1) {
            return computeJValue(jind - 1, dr, M, width, jmax, 1) * expm1 +
                computeJValue(jind, dr, M, width, jmax, 2) * expm;
        } else {
            return computeJValue(jind - 1, dr, M, width, jmax, 1) * expm1 +
                computeJValue(jind, dr, M, width, jmax, 2) * expm +
                computeJValue(jind + 1, dr, M, width, jmax, 3) * expp1;
        }
    } else {
        if (j == -jhigh) {
            return computeJValue(jind, dr, M, width, jmax, 3) * expm +
                computeJValue(jind + 1, dr, M, width, jmax, 3) * expp1;
        } else if (j == -jhigh + 1) {
            return computeJValue(jind - 1, dr, M, width, jmax, 2) * expm1 +
                computeJValue(jind, dr, M, width, jmax, 2) * expm +
                computeJValue(jind + 1, dr, M, width, jmax, 3) * expp1;
                    
        } else if (j == jhigh) {
            return computeJValue(jind - 1, dr, M, width, jmax, 1) * expm1 +
                computeJValue(jind, dr, M, width, jmax, 1) * expm;
        } else if (j == jhigh - 1) {
            return computeJValue(jind - 1, dr, M, width, jmax, 1) * expm1 +
                computeJValue(jind, dr, M, width, jmax, 2) * expm +
                computeJValue(jind + 1, dr, M, width, jmax, 2) * expp1;
                    
        } else {
            return ((j == -jhigh + 2) ? computeJValue(jind - 2, dr, M, width, jmax, 1) * (isCoalesced ? 
                    *getArrayAt(jind - 2, QsAll, totalCount, idx) * exp(-(alpha + computeJValue(jind - 2, dr, M, width, jmax, 0)) * dt) : 
                    QsAll[jind - 2] * exp(-(alpha + computeJValue(jind - 2, dr, M, width, jmax, 0)) * dt))
                : zero) +
                computeJValue(jind - 1, dr, M, width, jmax, 1) * expm1 +
                computeJValue(jind, dr, M, width, jmax, 2) * expm +
                computeJValue(jind + 1, dr, M, width, jmax, 3) * expp1 +
                ((j == jhigh - 2) ? computeJValue(jind + 2, dr, M, width, jmax, 3) * (isCoalesced ? 
                    *getArrayAt(jind + 2, QsAll, totalCount, idx) * exp(-(alpha + computeJValue(jind + 2, dr, M, width, jmax, 0)) * dt) : 
                    QsAll[jind + 2] * exp(-(alpha + computeJValue(jind + 2, dr, M, width, jmax, 0)) * dt))
                : zero);
        }
    }
}

/*
 * Backward helper using gather
 */
__device__ inline real backward_helper(
    const int idx,
    const int j, 
    const int jind,
    const real dr, 
    const real dt, 
    const real M, 
    const int width, 
    const int jmax,
    real* QsAll,
    const int totalCount,
    const real alpha,
    bool isCoalesced) 
{
    auto callExp = exp(-(alpha + computeJValue(jind, dr, M, width, jmax, 0)) * dt);

    if (j == jmax)
    {
        // Top edge branching
        return (computeJValue(jind, dr, M, width, jmax, 1) * (isCoalesced ? *getArrayAt(jind, QsAll, totalCount, idx) : QsAll[jind]) +
            computeJValue(jind, dr, M, width, jmax, 2) * (isCoalesced ? *getArrayAt(jind - 1, QsAll, totalCount, idx) : QsAll[jind - 1]) +
            computeJValue(jind, dr, M, width, jmax, 3) * (isCoalesced ? *getArrayAt(jind - 2, QsAll, totalCount, idx) : QsAll[jind - 2])) *
            callExp;
    }
    else if (j == -jmax)
    {
        // Bottom edge branching
        return (computeJValue(jind, dr, M, width, jmax, 1) * (isCoalesced ? *getArrayAt(jind + 2, QsAll, totalCount, idx) : QsAll[jind + 2]) +
            computeJValue(jind, dr, M, width, jmax, 2) * (isCoalesced ? *getArrayAt(jind + 1, QsAll, totalCount, idx) : QsAll[jind + 1]) +
            computeJValue(jind, dr, M, width, jmax, 3) * (isCoalesced ? *getArrayAt(jind, QsAll, totalCount, idx) : QsAll[jind])) *
            callExp;
    }
    else
    {
        // Standard branching
        return (computeJValue(jind, dr, M, width, jmax, 1) * (isCoalesced ? *getArrayAt(jind + 1, QsAll, totalCount, idx) : QsAll[jind + 1]) +
            computeJValue(jind, dr, M, width, jmax, 2) * (isCoalesced ? *getArrayAt(jind, QsAll, totalCount, idx) : QsAll[jind]) +
            computeJValue(jind, dr, M, width, jmax, 3) * (isCoalesced ? *getArrayAt(jind - 1, QsAll, totalCount, idx) : QsAll[jind - 1])) *
            callExp;
    }
}

__global__ void
kernelPaddingPerThreadBlock(real *res, const OptionConstants *options, real *QsAll, real *QsCopyAll, real *alphasAll, int *MaxWidths, int *MaxHeights, const int totalCount, const int blockSize, const int yieldCurveSize)
{
    const int bidx = blockIdx.x;
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Out of options check
    if (idx >= totalCount) return;

    auto c = options[idx];
    auto alpha = getYieldAtYear(c.dt, c.termUnit, YieldCurve, yieldCurveSize);
    *getUnpaddedArrayAt(c.jmax, QsAll, totalCount, idx, bidx, blockSize, MaxWidths) = one;
    *getUnpaddedArrayAt(c.jmax, alphasAll, totalCount, idx, bidx, blockSize, MaxHeights) = alpha;

    for (auto i = 1; i <= c.n; ++i)
    {
        const auto jhigh = min(i, c.jmax);
        real alpha_val = 0;

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - (-c.jmax); // array index for j
            
            // real Q = forward_helper(idx, i, j, jind, jhigh, c.dr, c.dt, c.M, c.width, c.jmax, QsAll, totalCount, alpha, true);            
            real Q = zero;
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            *getUnpaddedArrayAt(jind, QsCopyAll, totalCount, idx, bidx, blockSize, MaxWidths) = Q;
            alpha_val += Q * exp(-computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0) * c.dt);
        }

        alpha = computeAlpha(alpha_val, i-1, c.dt, c.termUnit, YieldCurve, yieldCurveSize);
        *getUnpaddedArrayAt(i, alphasAll, totalCount, idx, bidx, blockSize, MaxHeights) = alpha;

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
        auto alpha = *getUnpaddedArrayAt(i, alphasAll, totalCount, idx, bidx, blockSize, MaxHeights);
        
        // auto isMaturity = i == ((int)(c.t / c.dt));

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            // after obtaining the result from (i+1) nodes, set the call for ith node
            // *getArrayAt(jind, QsCopyAll, totalCount, idx) = computeCallValue(isMaturity, c, 
            //     backward_helper(idx, j, jind, c.dr, c.dt, c.M, c.width, c.jmax, QsAll, totalCount, alpha, true));
            *getUnpaddedArrayAt(jind, QsCopyAll, totalCount, idx, bidx, blockSize, MaxWidths) = zero;

        }

        // Switch call arrays
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;

        fillArrayColumn(c.width, 0, QsCopyAll, totalCount, idx);
    }

    res[idx] = *getUnpaddedArrayAt(c.jmax, QsAll, totalCount, idx, bidx, blockSize, MaxWidths);
}

void computeOptionsWithPaddingPerThreadBlock(const vector<OptionConstants> &options, const vector<Yield> &yield, vector<real> &results, bool isTest = false)
{
    const auto count = options.size();
    const auto blockSize = 64;
    const auto blockCount = ceil(count / ((float)blockSize));

    // Compute padding
    vector<int> maxWidths(blockCount, 0);
    vector<int> maxHeights(blockCount, 0);
    int blockCounter = 0;
    
    for (int i = 0; i < options.size(); ++i) {
        auto &option = options.at(i);
        if (option.width > maxWidths.at(blockCounter))
        {
            maxWidths.at(blockCounter) = option.width;
        }
        if (option.n >= maxHeights.at(blockCounter))
        {
            maxHeights.at(blockCounter) = option.n + 1; // n + 1 alphas
        }   

        if ((i+1) % blockSize == 0) {
            blockCounter++;
        }     
    }
    
    int totalQsCount = 0;
    int totalAlphasCount = 0;

    // todo: this can maybe be done better in c++ ? :D 
    for (auto& n : maxWidths) {
        totalQsCount += n;
    }

    for (auto& n : maxHeights) {
        totalAlphasCount += n;
    }

    if (isTest)
    {
        int memorySize = count * sizeof(real) + count * sizeof(OptionConstants)
                        + 2 * totalQsCount * sizeof(real) + totalAlphasCount * sizeof(real);
        cout << "Running trinomial option pricing for " << count << " options with block size " << blockSize << endl;
        cout << "Global memory size " << memorySize / (1024.0 * 1024.0) << " MB" << endl;
    }

    const auto time_begin = steady_clock::now();

    real *d_result, *d_Qs, *d_QsCopy, *d_alphas;
    int *d_MaxWidths, *d_MaxHeights;
    OptionConstants *d_options;
    CudaSafeCall(cudaMalloc((void **)&d_result, count * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_options, count * sizeof(OptionConstants)));
    CudaSafeCall(cudaMalloc((void **)&d_Qs, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_QsCopy, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_MaxWidths, blockCount * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_MaxHeights, blockCount * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_alphas, totalAlphasCount * sizeof(real)));

    CudaSafeCall(cudaMemcpyToSymbol(YieldCurve, yield.data(), yield.size() * sizeof(Yield)));
    CudaSafeCall(cudaMemcpy(d_options, options.data(), count * sizeof(OptionConstants), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_MaxWidths, &maxWidths, blockCount * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_MaxHeights, &maxHeights, blockCount * sizeof(int), cudaMemcpyHostToDevice));

    auto time_begin_kernel = steady_clock::now();
    kernelPaddingPerThreadBlock<<<blockCount, blockSize>>>(d_result, d_options, d_Qs, d_QsCopy, d_alphas, d_MaxWidths, d_MaxHeights, count, blockSize, yield.size());
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
    cudaFree(d_MaxWidths);
    cudaFree(d_MaxHeights);

    auto time_end = steady_clock::now();
    if (isTest)
    {
        cout << "Kernel executed in " << duration_cast<milliseconds>(time_end_kernel - time_begin_kernel).count() << " ms" << endl;
        cout << "Total GPU time: " << duration_cast<milliseconds>(time_end - time_begin).count() << " ms" << endl
            << endl;
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
            
            real Q = forward_helper(idx, i, j, jind, jhigh, c.dr, c.dt, c.M, c.width, c.jmax, QsAll, totalCount, alpha, true);
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

            // after obtaining the result from (i+1) nodes, set the call for ith node
            *getArrayAt(jind, QsCopyAll, totalCount, idx) = computeCallValue(isMaturity, c, 
                backward_helper(idx, j, jind, c.dr, c.dt, c.M, c.width, c.jmax, QsAll, totalCount, alpha, true));
        }

        // Switch call arrays
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;

        fillArrayColumn(c.width, 0, QsCopyAll, totalCount, idx);
    }

    res[idx] = *getArrayAt(c.jmax, QsAll, totalCount, idx);
}

void computeOptionsCoalesced(const vector<OptionConstants> &options, const vector<Yield> &yield, vector<real> &results, bool isTest = false)
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

    CudaSafeCall(cudaMemcpyToSymbol(YieldCurve, yield.data(), yield.size() * sizeof(Yield)));
    CudaSafeCall(cudaMemcpy(d_options, options.data(), count * sizeof(OptionConstants), cudaMemcpyHostToDevice));

    auto time_begin_kernel = steady_clock::now();
    kernelCoalesced<<<blockCount, blockSize>>>(d_result, d_options, d_Qs, d_QsCopy, d_alphas, count, yield.size());
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


__global__ void
kernelNaive(real *res, OptionConstants *options, real *QsAll, real *QsCopyAll, real *alphasAll, 
    int *QsInd, int *alphasInd, int totalCount, int yieldCurveSize)
{
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Out of options check
    if (idx >= totalCount) return;

    auto c = options[idx];
    auto Qs = QsAll + QsInd[idx];
    auto QsCopy = QsCopyAll + QsInd[idx];
    auto alphas = alphasAll + alphasInd[idx];
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
            auto jind = j + c.jmax;      // array index for j            
            real Q = forward_helper(idx, i, j, jind, jhigh, c.dr, c.dt, c.M, c.width, c.jmax, Qs, totalCount, alpha, false);            

            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            QsCopy[jind] = Q; 
            alpha_val += Q * exp(-computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0) * c.dt);
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
            auto jind = j + c.jmax;      // array index for j

            // after obtaining the result from (i+1) nodes, set the call for ith node
            callCopy[jind] = computeCallValue(isMaturity, c, 
                backward_helper(idx, j, jind, c.dr, c.dt, c.M, c.width, c.jmax, call, totalCount, alpha, false));
        }

        // Switch call arrays
        auto callT = call;
        call = callCopy;
        callCopy = callT;

        thrust::fill_n(thrust::device, callCopy, c.width, 0);
    }

    res[idx] = call[c.jmax];
}

void computeOptionsNaive(const vector<OptionConstants> &options, const vector<Yield> &yield, vector<real> &results, bool isTest = false)
{
    // Compute indices
    const auto count = options.size();
    int* QsInd = new int[count];
    int* alphasInd = new int[count];
    QsInd[0] = 0;
    alphasInd[0] = 0;
    int totalQsCount = 0;
    int totalAlphasCount = 0;
    for (auto i = 0; i < count - 1; ++i)
    {
        auto &option = options.at(i);
        totalQsCount += option.width;
        totalAlphasCount += option.n + 1;
        QsInd[i + 1] = totalQsCount;
        alphasInd[i + 1] = totalAlphasCount;
    }
    totalQsCount += options.at(count - 1).width;
    totalAlphasCount += options.at(count - 1).n + 1;
    
    auto blockSize = 64;
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
    CudaSafeCall(cudaMemcpy(d_options, options.data(), count * sizeof(OptionConstants), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_QsInd, QsInd, count * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_alphasInd, alphasInd, count * sizeof(int), cudaMemcpyHostToDevice));

    auto time_begin_kernel = steady_clock::now();
    kernelNaive<<<blockCount, blockSize>>>(d_result, d_options, d_Qs, d_QsCopy, d_alphas, d_QsInd, d_alphasInd, count, yield.size());
    cudaThreadSynchronize();
    auto time_end_kernel = steady_clock::now();

    CudaCheckError();

    // Copy result
    cudaMemcpy(results.data(), d_result, count * sizeof(real), cudaMemcpyDeviceToHost);

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
